#!/usr/bin/env python3
"""
Quick evaluation script for a FastTD3 policy in Isaac-Lab.
Only Isaac-Lab is touched – other back-ends are ignored.
"""

import argparse
import numpy as np
import torch
import os

# --- your own helpers --------------------------------------------------------
from fast_td3 import load_policy
from fast_td3.environments.isaaclab_env import IsaacLabEnv
# -----------------------------------------------------------------------------


torch.no_grad()
def export_to_onnx(checkpoint_path: str, overwrite: bool = False) -> str:
    """
    Export the actor found in `checkpoint_path` to ONNX and save it next to the
    checkpoint in a sub-folder called `exported/policy.onnx`.

    Returns
    -------
    str
        Absolute path of the written ONNX file.
    """
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    export_dir = os.path.join(ckpt_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)

    onnx_path = os.path.join(export_dir, "policy.onnx")
    if os.path.exists(onnx_path) and not overwrite:
        print(f"[ONNX] File already exists: {onnx_path}")
        return onnx_path

    print(f"[ONNX] Exporting policy to {onnx_path} …")

    policy = load_policy(checkpoint_path)
    policy.eval()            # just to be sure
    example_inputs = torch.ones(1, policy.n_obs)

    # New torch-2.* style export – returns an OnnxProgram
    onnx_program = torch.onnx.export(
        policy,
        (example_inputs,),
        input_names=["obs"],
        output_names=["actions"],
        dynamo=True,          # <- uses torch-dynamo under the hood
    )
    onnx_program.optimize()   # optional graph clean-up
    onnx_program.save(onnx_path)

    print("[ONNX] Done.")
    return onnx_path


@torch.no_grad()
def rollout(policy, env, num_episodes: int = 10, device: torch.device = None):
    """Runs `num_episodes` episodes and returns a list of total rewards."""
    returns = []

    for ep in range(num_episodes):
        # Isaac-Lab permits a deterministic start with random_start_init=False
        obs = env.reset(random_start_init=False)
        done = torch.zeros(env.num_envs, dtype=torch.bool).to(device)
        ep_ret = torch.zeros(env.num_envs).to(device)

        while not done.all():
            # Policy expects torch tensors on the same device as itself
            actions = policy(obs)
            obs, reward, done, _ = env.step(actions)
            ep_ret += reward

        returns.append(ep_ret.mean().item())
        print(f"Episode {ep:02d}  return = {returns[-1]:.2f}")

    print(f"\nAverage return over {num_episodes} episodes: {np.mean(returns):.2f}")
    return returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to *.pt file produced during training")
    parser.add_argument("--env_name", required=True,
                        help="Isaac-Lab gym ID, e.g. 'Unitree-Go2-Velocity-Safe'")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Parallel environments to launch in Isaac-Lab")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of roll-outs to run")
    parser.add_argument("--render", action="store_true",
                        help="Render with Isaac-Sim GUI (slow)")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    device = torch.device(
        "cpu" if not torch.cuda.is_available() else "cuda:0"
    )
    print("Using device:", device)

    # Instantiate environment
    env = IsaacLabEnv(
        task_name=args.env_name,
        device=device.type,
        num_envs=args.num_envs,
        seed=0,
        headless=False,
    )
    # Load policy
    policy = load_policy(args.checkpoint).to(device)
    policy.eval()
    
    # Export policy to ONNX
    export_to_onnx(args.checkpoint, overwrite=True)

    # Run evaluation
    rollout(policy, env, num_episodes=args.episodes, device=device)


if __name__ == "__main__":
    main()