import argparse

import torch

from sim2mujoco.envs import make_env
from sim2mujoco.policy import load_policy

__all__ = ["evaluate"]


def evaluate(
    env_name: str,
    checkpoint: str,
    episodes: int = 1,
    device: str = "cpu",
    deterministic: bool = False,
    render: bool = True,
):
    """Run *episodes* of the MuJoCo *env_name* with a policy from *checkpoint*.

    The function blocks until all episodes finish and prints episode rewards to
    stdout. It is also import-friendly, so other scripts can call it directly.
    """

    env = make_env(env_name, render=render)

    # Sample one observation to infer sizes if needed
    obs_sample, _ = env.reset()

    # Try to locate YAML sibling for rl_games
    cfg_guess = None
    import os, glob
    if checkpoint.endswith(".pth"):
        candidates = glob.glob(os.path.join(os.path.dirname(checkpoint), "*.yml")) + glob.glob(os.path.join(os.path.dirname(checkpoint), "*.yaml"))
        if candidates:
            cfg_guess = candidates[0]

    actor_or_dict = load_policy(checkpoint, device=device, cfg_path=cfg_guess, env_key=env_name.lower())

    # If we only received a raw state_dict, try to rebuild a generic MLP
    import torch
    if isinstance(actor_or_dict, dict):
        from sim2mujoco.policy import build_mlp_from_state_dict

        # Observation/action dimensions – fall back to sample if spaces missing
        try:
            obs_dim = env.observation_space.shape[0]  # type: ignore
        except AttributeError:
            obs_dim = obs_sample.shape[-1]

        try:
            act_dim = env.action_space.shape[0]  # type: ignore
        except AttributeError:
            act_dim = env.model.nu  # number of actuators in MuJoCo model

        actor = build_mlp_from_state_dict(actor_or_dict, obs_dim, act_dim, device)
    else:
        actor = actor_or_dict

    # ------------------------------------------------------------------
    #  Info banner
    # ------------------------------------------------------------------

    def _space_info(space):
        try:
            return f"shape={space.shape}, dtype={getattr(space, 'dtype', '?')}"
        except Exception:
            return str(space)

    import textwrap

    total_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    print("=" * 60)
    print("sim2mujoco evaluation session")
    print("Env           :", env_name)
    try:
        print("Obs space     :", _space_info(env.observation_space))
        print("Act space     :", _space_info(env.action_space))
    except AttributeError:
        pass
    if hasattr(env, 'dt'):
        print("Simulation dt :", env.dt)
    print("Actor network :", actor.__class__.__name__)
    print("Total params  :", total_params)
    print("Model architecture:\n" + textwrap.indent(str(actor), "  "))
    print("Device        :", device)
    print("Episodes      :", episodes)
    print("Deterministic :", deterministic)
    print("=" * 60)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        reward_sum = 0.0

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                # SHAC actors accept an optional *deterministic* kwarg.
                try:
                    action_tensor = actor(obs_tensor, deterministic=deterministic)
                except TypeError:
                    # Fallback for actors without the kwarg
                    action_tensor = actor(obs_tensor)

            action = action_tensor.detach().cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            print(obs)
            print(reward)
            print(terminated)
            print(truncated)
            print(info)
            reward_sum += float(reward)
            done = terminated or truncated

        print(f"Episode {ep + 1}/{episodes} │ Reward: {reward_sum:.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained policy inside a MuJoCo environment."
    )
    parser.add_argument("--env", required=True, help="Environment key, e.g. 'ant'.")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the saved policy checkpoint."
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run the policy on (e.g. 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use the mean action instead of sampling (if supported).",
    )
    args = parser.parse_args()

    evaluate(
        env_name=args.env,
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        device=args.device,
        deterministic=args.deterministic,
        render=True,
    )
