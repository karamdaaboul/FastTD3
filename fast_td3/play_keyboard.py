#!/usr/bin/env python3
"""
Play a Fast-TD3 locomotion policy in an Isaac-Lab task and steer it
interactively with the keyboard (Se2Keyboard).

Keys (default bindings)

    arrow keys: +vx  / –vx
    arrow keys: +vy  / –vy
    y, x: +ωz  / –ωz
"""

import argparse
import signal
import sys
import torch

# -------------------------------------------------------------------- #
#  project-specific helpers                                            #
# -------------------------------------------------------------------- #
from fast_td3 import load_policy
from fast_td3.environments.isaaclab_env import IsaacLabEnv
# -------------------------------------------------------------------- #

# -------------------------------- utility --------------------------- #


def graceful_exit(signum, frame):
    print("\n[INFO] Stopping …")
    sys.exit(0)


@torch.no_grad()
def play_loop(policy, env, teleop, num_episodes: int = 10,
              device: torch.device = None):

    """
    Runs the environment until Ctrl-C (or max_steps) is reached.
    Each time-step we:
      1. read a new velocity command from the keyboard,
      2. write it into `env.cfg.velocity_command`,
      3. query the policy for actions,
      4. step the simulation.
    """

    obs = env.reset(random_start_init=False)
    teleop.reset()

    returns = []

    for ep in range(num_episodes):
        obs = env.reset(random_start_init=False)
        done = torch.zeros(env.num_envs, dtype=torch.bool).to(device)
        ep_ret = torch.zeros(env.num_envs).to(device)

        while not done.all():
            # Policy expects torch tensors on the same device as itself
            actions = policy(obs)
            obs, reward, done, _ = env.step(actions)
            # print the command part of the obs
            env.envs.unwrapped.cfg.velocity_command = teleop.advance() # (num_envs, 3)

            ep_ret += reward

        returns.append(ep_ret.mean().item())
        print(f"Episode {ep:02d}  return = {returns[-1]:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to *.pt policy checkpoint")
    parser.add_argument("--env_name", required=True,
                        help="Isaac-Lab gym ID, e.g. "
                             "Unitree-Go2-Velocity-Safe-Keyboard")
    parser.add_argument("--headless", action="store_true",
                        help="Run Isaac-Sim in headless mode")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of roll-outs to run")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, graceful_exit)   # Ctrl-C handler

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # ---------------- Environment ----------------
    env = IsaacLabEnv(
        task_name=args.env_name,
        device=device.type,
        num_envs=1,
        seed=0,
        headless=args.headless,
    )

    # ---------------- Policy ---------------------
    policy = load_policy(args.checkpoint).to(device)
    policy.eval()

    # ---------------- Keyboard -------------------
    from isaaclab.devices import Se2Keyboard            # keyboard tele-op
    teleop = Se2Keyboard()

    # ---------------- Play! ----------------------
    play_loop(policy, env, teleop, num_episodes=args.episodes, device=device)


if __name__ == "__main__":
    main()
