import os
import sys

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import random
import time
import math
import tqdm
import wandb
import numpy as np

try:
    # Required for avoiding IsaacGym import error
    import isaacgym
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tensordict import TensorDict
from fast_td3.fast_td3_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    PerTaskRewardNormalizer,
    mark_step,
)
from fast_td3.fast_td3_utils_safe import SimpleSafeReplayBuffer, save_params
from fast_td3.hyperparams import get_args
from fast_td3.lagrange import TD3LagrangianController

torch.set_float32_matmul_precision("high")

try:
    import jax.numpy as jnp
except ImportError:
    pass

def main():
    args = get_args()
    print(args)
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"

    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "mps" if args.cuda and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    if args.use_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{args.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    # Environment setup (unchanged - keeping your existing environment code)
    if args.env_name.startswith("h1hand-") or args.env_name.startswith("h1-"):
        from environments.humanoid_bench_env import HumanoidBenchEnv

        env_type = "humanoid_bench"
        envs = HumanoidBenchEnv(args.env_name, args.num_envs, device=device)
        eval_envs = envs
        render_env = HumanoidBenchEnv(
            args.env_name, 1, render_mode="rgb_array", device=device
        )
    elif args.env_name.startswith("Isaac-") or args.env_name.startswith("Unitree-"):
        from fast_td3.environments.isaaclab_env import IsaacLabEnv

        env_type = "isaaclab"
        envs = IsaacLabEnv(
            args.env_name,
            device.type,
            args.num_envs,
            args.seed,
            action_bounds=args.action_bounds,
        )
        eval_envs = envs
        render_env = envs
    elif args.env_name.startswith("MTBench-"):
        from fast_td3.environments.mtbench_env import MTBenchEnv

        env_name = "-".join(args.env_name.split("-")[1:])
        env_type = "mtbench"
        envs = MTBenchEnv(env_name, args.device_rank, args.num_envs, args.seed)
        eval_envs = envs
        render_env = envs
    else:
        from fast_td3.environments.mujoco_playground_env import make_env

        env_type = "mujoco_playground"
        envs, eval_envs, render_env = make_env(
            args.env_name,
            args.seed,
            args.num_envs,
            args.num_eval_envs,
            args.device_rank,
            use_tuned_reward=args.use_tuned_reward,
            use_domain_randomization=args.use_domain_randomization,
            use_push_randomization=args.use_push_randomization,
        )

    n_act = envs.num_actions
    n_obs = envs.num_obs if isinstance(envs.num_obs, int) else envs.num_obs[0]
    if envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs
            if isinstance(envs.num_privileged_obs, int)
            else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs
    action_low, action_high = -3.0, 3.0

    # Normalization setup (unchanged)
    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(
            shape=n_critic_obs, device=device
        )
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    if args.reward_normalization:
        if env_type in ["mtbench"]:
            reward_normalizer = PerTaskRewardNormalizer(
                num_tasks=envs.num_tasks,
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
        else:
            reward_normalizer = RewardNormalizer(
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
    else:
        reward_normalizer = nn.Identity()

    # Model setup (unchanged)
    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": args.num_envs,
        "device": device,
        "init_scale": args.init_scale,
        "hidden_dim": args.actor_hidden_dim,
    }
    critic_kwargs = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "hidden_dim": args.critic_hidden_dim,
        "device": device,
    }

    if env_type == "mtbench":
        actor_kwargs["n_obs"] = n_obs - envs.num_tasks + args.task_embedding_dim
        critic_kwargs["n_obs"] = n_critic_obs - envs.num_tasks + args.task_embedding_dim
        actor_kwargs["num_tasks"] = envs.num_tasks
        actor_kwargs["task_embedding_dim"] = args.task_embedding_dim
        critic_kwargs["num_tasks"] = envs.num_tasks
        critic_kwargs["task_embedding_dim"] = args.task_embedding_dim

    if args.agent == "fasttd3_safe":
        if env_type in ["mtbench"]:
            from fast_td3 import MultiTaskActor, MultiTaskCritic
            actor_cls = MultiTaskActor
            critic_cls = MultiTaskCritic
        else:
            from fast_td3 import Actor, Critic
            actor_cls = Actor
            critic_cls = Critic
        print("Using FastTD3")
    elif args.agent == "fasttd3_simbav2":
        if env_type in ["mtbench"]:
            from fast_td3_simbav2 import MultiTaskActor, MultiTaskCritic
            actor_cls = MultiTaskActor
            critic_cls = MultiTaskCritic
        else:
            from fast_td3_simbav2 import Actor, Critic
            actor_cls = Actor
            critic_cls = Critic
        print("Using FastTD3 + SimbaV2")
        actor_kwargs.pop("init_scale")
        actor_kwargs.update({
            "scaler_init": math.sqrt(2.0 / args.actor_hidden_dim),
            "scaler_scale": math.sqrt(2.0 / args.actor_hidden_dim),
            "alpha_init": 1.0 / (args.actor_num_blocks + 1),
            "alpha_scale": 1.0 / math.sqrt(args.actor_hidden_dim),
            "expansion": 4,
            "c_shift": 3.0,
            "num_blocks": args.actor_num_blocks,
        })
        critic_kwargs.update({
            "scaler_init": math.sqrt(2.0 / args.critic_hidden_dim),
            "scaler_scale": math.sqrt(2.0 / args.critic_hidden_dim),
            "alpha_init": 1.0 / (args.critic_num_blocks + 1),
            "alpha_scale": 1.0 / math.sqrt(args.critic_hidden_dim),
            "num_blocks": args.critic_num_blocks,
            "expansion": 4,
            "c_shift": 3.0,
        })
    else:
        raise ValueError(f"Agent {args.agent} not supported")

    actor = actor_cls(**actor_kwargs)

    if env_type in ["mtbench"]:
        policy = actor.explore
    else:
        from tensordict import from_module
        actor_detach = actor_cls(**actor_kwargs)
        from_module(actor).data.to_module(actor_detach)
        policy = actor_detach.explore

    # Critics setup
    qnet = critic_cls(**critic_kwargs)
    qnet_target = critic_cls(**critic_kwargs)
    qnet_target.load_state_dict(qnet.state_dict())

    # Optimizers
    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )

    # Schedulers
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
    )

    # Safety critic setup
    critic_kwargs["v_min"] = args.v_safe_min
    critic_kwargs["v_max"] = args.v_safe_max
    safety_qnet = critic_cls(**critic_kwargs)
    safety_qnet_target = critic_cls(**critic_kwargs)
    safety_qnet_target.load_state_dict(safety_qnet.state_dict())

    safety_optimizer = optim.AdamW(
        safety_qnet.parameters(),
        lr=torch.tensor(args.safety_critic_lr, device=device),
        weight_decay=args.weight_decay,
    )
    safety_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        safety_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
    )

    # Lagrangian controller
    lagrangian_controller = TD3LagrangianController(
        cost_threshold=args.cost_threshold,
        kp=0.05,
        ki=0.0005,
        kd=0.1,
        lambda_lr=0.001,
        lambda_max=100.0,
        device=device,
    )

    # Replay buffer
    rb = SimpleSafeReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=envs.asymmetric_obs,
        playground_mode=env_type == "mujoco_playground",
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,
    )

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    def evaluate():
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        if env_type == "isaaclab":
            obs = eval_envs.reset(random_start_init=False)
        else:
            obs = eval_envs.reset()

        for i in range(eval_envs.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                obs = normalize_obs(obs, update=False)
                actions = actor(obs)

            next_obs, rewards, dones, infos = eval_envs.step(actions.float())

            if env_type == "mtbench":
                rewards = (
                    infos["episode"]["success"].float()
                    if "episode" in infos else 0.0
                )
            episode_returns = torch.where(
                ~done_masks, episode_returns + rewards, episode_returns
            )
            episode_lengths = torch.where(
                ~done_masks, episode_lengths + 1, episode_lengths
            )
            if env_type == "mtbench" and "episode" in infos:
                dones = dones | infos["episode"]["success"]
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        return episode_returns.mean().item(), episode_lengths.mean().item()

    def render_with_rollout():
        if env_type == "humanoid_bench":
            obs = render_env.reset()
            renders = [render_env.render()]
        elif env_type in ["isaaclab", "mtbench"]:
            raise NotImplementedError(
                "We don't support rendering for IsaacLab and MTBench environments"
            )
        else:
            obs = render_env.reset()
            render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            renders = [render_env.state]
        for i in range(render_env.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                obs = normalize_obs(obs, update=False)
                actions = actor(obs)
            next_obs, _, done, _ = render_env.step(actions.float())
            if env_type == "mujoco_playground":
                render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            if i % 2 == 0:
                if env_type == "humanoid_bench":
                    renders.append(render_env.render())
                else:
                    renders.append(render_env.state)
            if done.any():
                break
            obs = next_obs

        if env_type == "mujoco_playground":
            renders = render_env.render_trajectory(renders)
        return renders

    def update_main(data, logs_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = observations
                next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            if args.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )

            next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                action_low, action_high
            )
            discount = args.gamma ** data["next"]["effective_n_steps"]
            
            if args.critic_type == "distributional":
                with torch.no_grad():
                    qf1_next_target_projected, qf2_next_target_projected = (
                        qnet_target.projection(
                            next_critic_observations,
                            next_state_actions,
                            rewards,
                            bootstrap,
                            discount,
                        )
                    )
                    qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                    qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
                    if args.use_cdq:
                        qf_next_target_dist = torch.where(
                            qf1_next_target_value.unsqueeze(1)
                            < qf2_next_target_value.unsqueeze(1),
                            qf1_next_target_projected,
                            qf2_next_target_projected,
                        )
                        qf1_next_target_dist = qf2_next_target_dist = (
                            qf_next_target_dist
                        )
                    else:
                        qf1_next_target_dist, qf2_next_target_dist = (
                            qf1_next_target_projected,
                            qf2_next_target_projected,
                        )

                qf1, qf2 = qnet(critic_observations, actions)
                qf1_loss = -torch.sum(
                    qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
                ).mean()
                qf2_loss = -torch.sum(
                    qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
                ).mean()
            else:
                with torch.no_grad():
                    q1_next, q2_next = qnet_target(
                        next_critic_observations, next_state_actions
                    )
                    q1_next_val = qnet_target.get_value(q1_next)
                    q2_next_val = qnet_target.get_value(q2_next)
                    if args.use_cdq:
                        target_q = torch.minimum(q1_next_val, q2_next_val)
                    else:
                        target_q = 0.5 * (q1_next_val + q2_next_val)
                    target_q = rewards + bootstrap * discount * target_q

                q1_pred, q2_pred = qnet(critic_observations, actions)
                q1_pred_val = qnet.get_value(q1_pred)
                q2_pred_val = qnet.get_value(q2_pred)
                qf1_loss = F.mse_loss(q1_pred_val, target_q)
                qf2_loss = F.mse_loss(q2_pred_val, target_q)
                
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if args.use_grad_norm_clipping:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()

        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf1_next_target_value.max().detach()
        logs_dict["qf_min"] = qf1_next_target_value.min().detach()
        return logs_dict
    
    def update_safety(data, logs_dict):
        """
        Distributional TD-backup for the cost critic Q_C.
        """
        with autocast(device_type=amp_device_type,
                    dtype=amp_dtype,
                    enabled=amp_enabled):
            observations = data["observations"]
            next_observations = data["next"]["observations"]

            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = observations
                next_critic_observations = next_observations

            actions = data["actions"]
            costs = data["next"]["costs"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions).mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )
            next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                action_low, action_high
            )
            discount = args.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                q1_next_tgt_proj, q2_next_tgt_proj = safety_qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    costs,
                    bootstrap,
                    discount,
                )
                q1_val = safety_qnet_target.get_value(q1_next_tgt_proj)
                q2_val = safety_qnet_target.get_value(q2_next_tgt_proj)
                if args.use_cdq:
                    q_next_tgt_dist = torch.where(
                        q1_val.unsqueeze(1) < q2_val.unsqueeze(1),
                        q1_next_tgt_proj,
                        q2_next_tgt_proj,
                    )
                    q1_next_tgt_dist = q2_next_tgt_dist = q_next_tgt_dist
                else:
                    q1_next_tgt_dist, q2_next_tgt_dist = (
                        q1_next_tgt_proj,
                        q2_next_tgt_proj,
                    )

            q1, q2 = safety_qnet(critic_observations, actions)
            q1_loss = -torch.sum(q1_next_tgt_dist * F.log_softmax(q1, dim=1), dim=1).mean()
            q2_loss = -torch.sum(q2_next_tgt_dist * F.log_softmax(q2, dim=1), dim=1).mean()
            safety_loss = q1_loss + q2_loss

        safety_optimizer.zero_grad(set_to_none=True)
        scaler.scale(safety_loss).backward()
        scaler.unscale_(safety_optimizer)

        if args.use_grad_norm_clipping:
            safety_grad_norm = torch.nn.utils.clip_grad_norm_(
                safety_qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            safety_grad_norm = torch.tensor(0.0, device=device)

        scaler.step(safety_optimizer)
        scaler.update()

        logs_dict["safety_loss"] = safety_loss.detach()
        logs_dict["safety_grad_norm"] = safety_grad_norm.detach()
        logs_dict["safety_qf_max"] = q1_val.max().detach()
        logs_dict["safety_qf_min"] = q1_val.min().detach()
        return logs_dict

    # FIXED: Separate policy update function that takes lambda as parameter
    def update_pol_constrained(data, lambda_value, logs_dict):
        """
        TD3 actor update with Lagrangian safety constraints
        Lambda value is passed as parameter (updated externally)
        """
        with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            critic_observations = (
                data["critic_observations"]
                if envs.asymmetric_obs
                else data["observations"]
            )

            # Get current actions from actor
            current_actions = actor(data["observations"])

            # Reward Q-values (existing TD3 approach)
            qf1, qf2 = qnet(critic_observations, current_actions)
            if args.critic_type == "distributional":
                qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
                qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
            else:
                qf1_value = qnet.get_value(qf1)
                qf2_value = qnet.get_value(qf2)
                
            if args.use_cdq:
                reward_qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                reward_qf_value = (qf1_value + qf2_value) / 2.0

            # Safety Q-values (NEW for TD3-Lagrangian)
            safety_qf1, safety_qf2 = safety_qnet(critic_observations, current_actions)
            safety_qf1_value = safety_qnet.get_value(F.softmax(safety_qf1, dim=1))
            safety_qf2_value = safety_qnet.get_value(F.softmax(safety_qf2, dim=1))
            
            if args.use_cdq:
                safety_qf_value = torch.minimum(safety_qf1_value, safety_qf2_value)
            else:
                safety_qf_value = (safety_qf1_value + safety_qf2_value) / 2.0

            # TD3-Lagrangian Actor Loss: maximize reward, minimize expected cost
            # Use the lambda_value passed as parameter (no update inside this function)
            actor_loss = -reward_qf_value.mean() + lambda_value * safety_qf_value.mean()

        # Optimization
        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        
        if args.use_grad_norm_clipping:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)
            
        scaler.step(actor_optimizer)
        scaler.update()

        logs_dict["actor_loss"] = actor_loss.detach()
        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["safety_qf_value_mean"] = safety_qf_value.mean().detach()
        logs_dict["reward_qf_value_mean"] = reward_qf_value.mean().detach()
        
        return logs_dict

    # FIXED: Separate function to update lagrangian multiplier
    def update_lagrangian_multiplier(data, lagrangian_controller, logs_dict):
        """
        Update Lagrangian multiplier based on current cost data
        This runs outside the compiled policy update
        """
        # Get current episode cost for Lagrangian update
        current_episode_cost = data["next"]["costs"].mean().item()
        lambda_value = lagrangian_controller.update(current_episode_cost)
        
        # Add Lagrangian logs
        lagrangian_logs = lagrangian_controller.get_logs()
        logs_dict.update(lagrangian_logs)
        
        return lambda_value, logs_dict

    @torch.no_grad()
    def soft_update(src, tgt, tau: float):
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    # CRITICAL FIX: Exclude policy update from compilation to avoid CUDA graph issues
    if args.compile:
        compile_mode = args.compile_mode
        update_main = torch.compile(update_main, mode=compile_mode)
        update_safety = torch.compile(update_safety, mode=compile_mode)
        # DON'T compile the constrained policy update to avoid Lagrangian controller issues
        # update_pol_constrained = torch.compile(update_pol_constrained, mode=compile_mode)
        policy = torch.compile(policy, mode=None)
        normalize_obs = torch.compile(obs_normalizer.forward, mode=None)
        normalize_critic_obs = torch.compile(critic_obs_normalizer.forward, mode=None)
        if args.reward_normalization:
            update_stats = torch.compile(reward_normalizer.update_stats, mode=None)
        normalize_reward = torch.compile(reward_normalizer.forward, mode=None)
    else:
        normalize_obs = obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    # Environment initialization
    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = envs.reset()
        
    if args.checkpoint_path:
        torch_checkpoint = torch.load(
            f"{args.checkpoint_path}", map_location=device, weights_only=False
        )
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        critic_obs_normalizer.load_state_dict(
            torch_checkpoint["critic_obs_normalizer_state"]
        )
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        global_step = torch_checkpoint["global_step"]
    else:
        global_step = 0

    dones = None
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    start_time = None
    desc = ""

    # Main training loop
    while global_step < args.total_timesteps:
        mark_step()
        logs_dict = TensorDict()
        if (
            start_time is None
            and global_step >= args.measure_burnin + args.learning_starts
        ):
            start_time = time.time()
            measure_burnin = global_step

        with torch.no_grad(), autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            norm_obs = normalize_obs(obs)
            actions = policy(obs=norm_obs, dones=dones)

        next_obs, rewards, dones, infos = envs.step(actions.float())
        truncations = infos["time_outs"]

        if args.reward_normalization:
            if env_type == "mtbench":
                task_ids_one_hot = obs[..., -envs.num_tasks :]
                task_indices = torch.argmax(task_ids_one_hot, dim=1)
                update_stats(rewards, dones.float(), task_ids=task_indices)
            else:
                update_stats(rewards, dones.float())

        if envs.asymmetric_obs:
            next_critic_obs = infos["observations"]["critic"]
            
        true_next_obs = torch.where(
            dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        )
        if envs.asymmetric_obs:
            true_next_critic_obs = torch.where(
                dones[:, None] > 0,
                infos["observations"]["raw"]["critic_obs"],
                next_critic_obs,
            )

        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(
                        rewards, device=device, dtype=torch.float
                    ),
                    "costs": torch.as_tensor(
                        infos["cost"], device=device, dtype=torch.float
                    ),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        if envs.asymmetric_obs:
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs
        rb.extend(transition)

        obs = next_obs
        if envs.asymmetric_obs:
            critic_obs = next_critic_obs

        # Training updates
        if global_step > args.learning_starts:
            for i in range(args.num_updates):
                data = rb.sample(max(1, args.batch_size // args.num_envs))
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(
                    data["next"]["observations"]
                )
                if envs.asymmetric_obs:
                    data["critic_observations"] = normalize_critic_obs(
                        data["critic_observations"]
                    )
                    data["next"]["critic_observations"] = normalize_critic_obs(
                        data["next"]["critic_observations"]
                    )
                raw_rewards = data["next"]["rewards"]
                if env_type in ["mtbench"] and args.reward_normalization:
                    task_ids_one_hot = data["observations"][..., -envs.num_tasks :]
                    task_indices = torch.argmax(task_ids_one_hot, dim=1)
                    data["next"]["rewards"] = normalize_reward(
                        raw_rewards, task_ids=task_indices
                    )
                else:
                    data["next"]["rewards"] = normalize_reward(raw_rewards)

                logs_dict = update_main(data, logs_dict)
                logs_dict = update_safety(data, logs_dict)

                # CRITICAL FIX: Update Lagrangian multiplier OUTSIDE policy update
                lambda_value, logs_dict = update_lagrangian_multiplier(data, lagrangian_controller, logs_dict)

                # CRITICAL: Use constrained policy update with external lambda value
                if args.num_updates > 1:
                    if i % args.policy_frequency == 1:
                        logs_dict = update_pol_constrained(data, lambda_value, logs_dict)
                else:
                    if global_step % args.policy_frequency == 0:
                        logs_dict = update_pol_constrained(data, lambda_value, logs_dict)

                # CRITICAL: Update both target networks
                soft_update(qnet, qnet_target, args.tau)
                soft_update(safety_qnet, safety_qnet_target, args.tau)

            # Logging
            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "actor_loss": logs_dict["actor_loss"].mean(),
                        "qf_loss": logs_dict["qf_loss"].mean(),
                        "qf_max": logs_dict["qf_max"].mean(),
                        "qf_min": logs_dict["qf_min"].mean(),
                        "safety_qf_max": logs_dict["safety_qf_max"].mean(),
                        "safety_qf_min": logs_dict["safety_qf_min"].mean(),
                        "actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                        "critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                        "safety_loss": logs_dict["safety_loss"].mean(),
                        "safety_grad_norm": logs_dict["safety_grad_norm"].mean(),
                        "lagrangian_multiplier": logs_dict.get("lagrangian_multiplier", 0.0),
                        "constraint_violation": logs_dict.get("constraint_violation", 0.0),
                        "safety_qf_value_mean": logs_dict["safety_qf_value_mean"].mean(),
                        "reward_qf_value_mean": logs_dict["reward_qf_value_mean"].mean(),
                        "env_rewards": rewards.mean(),
                        "buffer_rewards": raw_rewards.mean(),
                        "env_costs": infos["cost"].mean(),
                        "buffer_costs": data["next"]["costs"].mean(),
                    }

                    if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                        print(f"Evaluating at global step {global_step}")
                        eval_avg_return, eval_avg_length = evaluate()
                        if env_type in ["humanoid_bench", "isaaclab", "mtbench"]:
                            obs = envs.reset()
                        logs["eval_avg_return"] = eval_avg_return
                        logs["eval_avg_length"] = eval_avg_length

                    if (
                        args.render_interval > 0
                        and global_step % args.render_interval == 0
                    ):
                        renders = render_with_rollout()
                        render_video = wandb.Video(
                            np.array(renders).transpose(0, 3, 1, 2),
                            fps=30,
                            format="gif",
                        )
                        logs["render_video"] = render_video
                        
                if args.use_wandb:
                    wandb.log(
                        {
                            "speed": speed,
                            "frame": global_step * args.num_envs,
                            "critic_lr": q_scheduler.get_last_lr()[0],
                            "actor_lr": actor_scheduler.get_last_lr()[0],
                            "safety_lr": safety_scheduler.get_last_lr()[0],
                            **logs,
                        },
                        step=global_step,
                    )

            if (
                args.save_interval > 0
                and global_step > 0
                and global_step % args.save_interval == 0
            ):
                print(f"Saving model at global step {global_step}")
                save_params(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    safety_qnet,
                    safety_qnet_target,
                    obs_normalizer,
                    critic_obs_normalizer,
                    args,
                    f"models/{run_name}_{global_step}.pt",
                )

        global_step += 1
        actor_scheduler.step()
        q_scheduler.step()
        safety_scheduler.step()
        pbar.update(1)

    save_params(
        global_step,
        actor,
        qnet,
        qnet_target,
        safety_qnet,
        safety_qnet_target,
        obs_normalizer,
        critic_obs_normalizer,
        args,
        f"models/{run_name}_final.pt",
    )

if __name__ == "__main__":
    main()
