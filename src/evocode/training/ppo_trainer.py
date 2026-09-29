"""PPO Trainer for EvoCode."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from evocode.policy.mlp import MLPPolicy, PolicyConfig
from evocode.environment.evocode_env import EvoCodeEnv
from evocode.training.ppo import PPOConfig


class RandomPolicy:
    """Random policy for baseline comparison."""

    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def predict(self, obs: np.ndarray) -> int:
        """Return random action."""
        return np.random.randint(0, len(self.action_space))

    def get_action_probs(self, obs: np.ndarray) -> np.ndarray:
        """Uniform distribution."""
        n = len(self.action_space)
        return np.ones(n) / n


class RolloutBuffer:
    """Buffer for storing rollouts."""

    def __init__(self, n_steps: int, observation_dim: int, device: torch.device) -> None:
        self.n_steps = n_steps
        self.observation_dim = observation_dim
        self.device = device

        self.observations = np.zeros((n_steps, observation_dim), dtype=np.float32)
        self.actions = np.zeros(n_steps, dtype=np.int64)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.bool_)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.ptr = 0
            self.full = True

    def get(self) -> dict[str, np.ndarray]:
        if self.full:
            return {
                "observations": self.observations,
                "actions": self.actions,
                "rewards": self.rewards,
                "values": self.values,
                "log_probs": self.log_probs,
                "dones": self.dones,
            }
        else:
            return {
                "observations": self.observations[:self.ptr],
                "actions": self.actions[:self.ptr],
                "rewards": self.rewards[:self.ptr],
                "values": self.values[:self.ptr],
                "log_probs": self.log_probs[:self.ptr],
                "dones": self.dones[:self.ptr],
            }

    def reset(self) -> None:
        self.ptr = 0
        self.full = False


class PPOTrainer:
    """PPO Trainer for EvoCode policies."""

    def __init__(
        self,
        policy: MLPPolicy,
        env: EvoCodeEnv,
        config: PPOConfig | None = None,
    ) -> None:
        self.policy = policy
        self.env = env
        self.config = config or PPOConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.buffer = RolloutBuffer(
            n_steps=self.config.n_steps,
            observation_dim=self.policy.config.observation_dim,
            device=self.device,
        )

        self.global_step = 0
        self.episode_count = 0

    def collect_rollouts(self) -> dict[str, np.ndarray]:
        """Collect rollouts from environment."""
        self.buffer.reset()
        obs, info = self.env.reset()

        while not self.buffer.full:
            # Get action from policy
            logits, value = self.policy.predict(obs.reshape(1, -1))
            probs = torch.softmax(torch.from_numpy(logits), dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                value=value.item(),
                log_prob=log_prob,
                done=done,
            )

            obs = next_obs
            self.global_step += 1

            if done:
                obs, info = self.env.reset()
                self.episode_count += 1

        return self.buffer.get()

    def compute_advantages(self, rollouts: dict[str, np.ndarray]) -> np.ndarray:
        """Compute GAE advantages."""
        rewards = rollouts["rewards"]
        values = rollouts["values"]
        dones = rollouts["dones"]

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for i in reversed(range(n)):
            if i == n - 1:
                next_value = 0.0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]

            delta = rewards[i] + self.config.gamma * next_value - values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae * (1.0 - dones[i])
            advantages[i] = gae

        return advantages

    def update_policy(
        self, rollouts: dict[str, np.ndarray], advantages: np.ndarray
    ) -> dict[str, float]:
        """Update policy using PPO."""
        # Convert to tensors
        obs = torch.from_numpy(rollouts["observations"]).float().to(self.device)
        actions = torch.from_numpy(rollouts["actions"]).long().to(self.device)
        old_log_probs = torch.from_numpy(rollouts["log_probs"]).float().to(self.device)
        returns = torch.from_numpy(advantages + rollouts["values"]).float().to(self.device)
        advantages_t = torch.from_numpy(advantages).float().to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Dataset
        dataset_size = len(obs)
        indices = np.arange(dataset_size)

        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(self.config.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.config.batch_size):
                end = min(start + self.config.batch_size, dataset_size)
                batch_idx = indices[start:end]

                # Forward pass
                logits, values = self.policy.forward(obs[batch_idx])
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                # Log probs
                new_log_probs = dist.log_prob(actions[batch_idx])
                entropy = dist.entropy().mean()

                # Ratio
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])

                # Surrogate losses
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = (
                    torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
                    * advantages_t[batch_idx]
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(), returns[batch_idx])

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
        }

    def train_step(self) -> dict[str, float]:
        """One training step: collect rollouts + update."""
        rollouts = self.collect_rollouts()
        advantages = self.compute_advantages(rollouts)
        loss = self.update_policy(rollouts, advantages)

        # Compute mean reward
        mean_reward = np.mean(rollouts["rewards"])

        return {
            **loss,
            "mean_reward": mean_reward,
            "episodes": self.episode_count,
        }

    def evaluate(self, n_episodes: int = 10) -> dict[str, float]:
        """Evaluate current policy."""
        rewards = []
        lengths = []
        successes = 0

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                logits, _ = self.policy.predict(obs.reshape(1, -1))
                action = logits.argmax(axis=-1).item()
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1

            rewards.append(episode_reward)
            lengths.append(episode_length)
            if info.get("success", False):
                successes += 1

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "success_rate": successes / n_episodes,
            "mean_episode_length": np.mean(lengths),
        }

    def save_checkpoint(self, path: Path) -> None:
        """Save checkpoint with policy, optimizer, and training state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "policy_config": self.policy.config,
                "ppo_config": self.config,
                "global_step": self.global_step,
                "episode_count": self.episode_count,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        """Load checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.episode_count = checkpoint.get("episode_count", 0)