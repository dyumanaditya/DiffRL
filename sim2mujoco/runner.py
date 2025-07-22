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
    actor = load_policy(checkpoint, device=device)

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

            obs, reward, terminated, truncated, _ = env.step(action)
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
