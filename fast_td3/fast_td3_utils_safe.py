import os

from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from tensordict import TensorDict
from fast_td3.fast_td3_utils import SimpleReplayBuffer

class SimpleSafeReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
    ):
        """
        A simple replay buffer that stores transitions in a circular buffer.
        Supports n-step returns and asymmetric observations.

        When playground_mode=True, critic_observations are treated as a concatenation of
        regular observations and privileged observations, and only the privileged part is stored
        to save memory.

        TODO (Younggyo): Refactor to split this into SimpleReplayBuffer and NStepReplayBuffer
        """
        super().__init__(n_env, buffer_size, n_obs, n_act, n_critic_obs, asymmetric_obs, playground_mode, n_steps, gamma, device)
        self.costs = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.float
        )
        self.next_costs = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.float
        )

    @torch.no_grad()
    def extend(
        self,
        tensor_dict: TensorDict,
    ):
        costs = tensor_dict["next"]["costs"]
        ptr = self.ptr % self.buffer_size
        self.costs[:, ptr] = costs
        self.next_costs[:, ptr] = costs
        super().extend(tensor_dict)

    @torch.no_grad()
    def extend(self, tensor_dict: TensorDict):
        """Extend the buffer with new transitions including costs."""
        costs = tensor_dict["next"]["costs"]
        ptr = self.ptr % self.buffer_size
        
        # Store costs at the current pointer position
        self.costs[:, ptr] = costs
        
        # Call parent extend to handle all other data
        super().extend(tensor_dict)

    @torch.no_grad()
    def sample(self, batch_size: int):
        """Sample transitions from the buffer, computing n-step returns for both rewards and costs."""
        
        if self.n_steps == 1:
            # Single-step case - simpler logic
            indices = torch.randint(
                0,
                min(self.buffer_size, self.ptr),
                (self.n_env, batch_size),
                device=self.device,
            )
            
            # Gather all the basic data
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
            
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            next_observations = torch.gather(
                self.next_observations, 1, obs_indices
            ).reshape(self.n_env * batch_size, self.n_obs)
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )

            rewards = torch.gather(self.rewards, 1, indices).reshape(
                self.n_env * batch_size
            )
            costs = torch.gather(self.costs, 1, indices).reshape(
                self.n_env * batch_size
            )
            dones = torch.gather(self.dones, 1, indices).reshape(
                self.n_env * batch_size
            )
            truncations = torch.gather(self.truncations, 1, indices).reshape(
                self.n_env * batch_size
            )
            effective_n_steps = torch.ones_like(dones)
            
            # Handle asymmetric observations
            if self.asymmetric_obs:
                if self.playground_mode:
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)
                    next_privileged_observations = torch.gather(
                        self.next_privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                    next_critic_observations = torch.cat(
                        [next_observations, next_privileged_observations], dim=1
                    )
                else:
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
                    next_critic_observations = torch.gather(
                        self.next_critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
                    
        else:
            # Multi-step case - more complex logic
            if self.ptr >= self.buffer_size:
                current_pos = self.ptr % self.buffer_size
                curr_truncations = self.truncations[:, current_pos - 1].clone()
                self.truncations[:, current_pos - 1] = torch.logical_not(
                    self.dones[:, current_pos - 1]
                )
                indices = torch.randint(
                    0, self.buffer_size, (self.n_env, batch_size), device=self.device,
                )
            else:
                max_start_idx = max(1, self.ptr - self.n_steps + 1)
                indices = torch.randint(
                    0, max_start_idx, (self.n_env, batch_size), device=self.device,
                )
            
            # Get base observations and actions
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
            
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )
            
            # Handle asymmetric observations for base observations
            if self.asymmetric_obs:
                if self.playground_mode:
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                else:
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)

            # Create sequential indices for n-step sequences
            seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
            all_indices = (indices.unsqueeze(-1) + seq_offsets) % self.buffer_size

            # Gather all rewards, costs, and terminal flags for n-step sequences
            all_rewards = torch.gather(
                self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_costs = torch.gather(
                self.costs.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_dones = torch.gather(
                self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_truncations = torch.gather(
                self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices,
            )

            # Create masks for values after first done
            all_dones_shifted = torch.cat(
                [torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2
            )
            done_masks = torch.cumprod(1.0 - all_dones_shifted, dim=2)
            effective_n_steps = done_masks.sum(2)

            # Create discount factors
            discounts = torch.pow(
                self.gamma, torch.arange(self.n_steps, device=self.device)
            )

            # Apply masks and discounts to both rewards and costs
            masked_rewards = all_rewards * done_masks
            masked_costs = all_costs * done_masks
            discounted_rewards = masked_rewards * discounts.view(1, 1, -1)
            discounted_costs = masked_costs * discounts.view(1, 1, -1)

            # Sum along the n_step dimension
            n_step_rewards = discounted_rewards.sum(dim=2)
            n_step_costs = discounted_costs.sum(dim=2)

            # Find final indices for next observations
            first_done = torch.argmax((all_dones > 0).float(), dim=2)
            first_trunc = torch.argmax((all_truncations > 0).float(), dim=2)
            
            no_dones = all_dones.sum(dim=2) == 0
            no_truncs = all_truncations.sum(dim=2) == 0
            
            first_done = torch.where(no_dones, self.n_steps - 1, first_done)
            first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)
            final_indices = torch.minimum(first_done, first_trunc)

            # Gather final next observations and terminal flags
            final_next_obs_indices = torch.gather(
                all_indices, 2, final_indices.unsqueeze(-1)
            ).squeeze(-1)
            
            final_next_observations = self.next_observations.gather(
                1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            )
            final_dones = self.dones.gather(1, final_next_obs_indices)
            final_truncations = self.truncations.gather(1, final_next_obs_indices)

            # Handle asymmetric next observations
            if self.asymmetric_obs:
                if self.playground_mode:
                    final_next_privileged_observations = (
                        self.next_privileged_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.privileged_obs_size
                            ),
                        )
                    )
                    next_privileged_observations = (
                        final_next_privileged_observations.reshape(
                            self.n_env * batch_size, self.privileged_obs_size
                        )
                    )
                    next_observations_reshaped = final_next_observations.reshape(
                        self.n_env * batch_size, self.n_obs
                    )
                    next_critic_observations = torch.cat(
                        [next_observations_reshaped, next_privileged_observations], dim=1,
                    )
                else:
                    final_next_critic_observations = (
                        self.next_critic_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.n_critic_obs
                            ),
                        )
                    )
                    next_critic_observations = final_next_critic_observations.reshape(
                        self.n_env * batch_size, self.n_critic_obs
                    )

            # Reshape everything to batch dimension
            rewards = n_step_rewards.reshape(self.n_env * batch_size)
            costs = n_step_costs.reshape(self.n_env * batch_size)
            dones = final_dones.reshape(self.n_env * batch_size)
            truncations = final_truncations.reshape(self.n_env * batch_size)
            effective_n_steps = effective_n_steps.reshape(self.n_env * batch_size)
            next_observations = final_next_observations.reshape(
                self.n_env * batch_size, self.n_obs
            )

        # Create output TensorDict with costs included
        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "costs": costs,  # Include costs in the output
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                    "effective_n_steps": effective_n_steps,
                },
            },
            batch_size=self.n_env * batch_size,
        )
        
        if self.asymmetric_obs:
            out["critic_observations"] = critic_observations
            out["next"]["critic_observations"] = next_critic_observations

        if self.n_steps > 1 and self.ptr >= self.buffer_size:
            # Roll back the truncation flags introduced for safe sampling
            self.truncations[:, current_pos - 1] = curr_truncations
            
        return out