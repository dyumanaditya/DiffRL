from .hopper import HopperMujocoEnv

try:
    import gymnasium as gym
    from gymnasium import error as gym_error
except ImportError as e:
    raise ImportError(
        "sim2mujoco requires Gymnasium >= 0.28.0 with MuJoCo support. "
        "Install it via `pip install 'gymnasium[mujoco]'` or "
        "`conda install -c conda-forge gymnasium mujoco`."
    ) from e

# Mapping from short environment keys to Gymnasium MuJoCo env IDs
MUJOCO_GYM_MAP = {
    # Classic MuJoCo control suite (gymnasium>=0.28)
    "ant": "Ant-v4",
    "hopper": "Hopper-v5",
    "halfcheetah": "HalfCheetah-v4",
    "humanoid": "Humanoid-v4",
    "walker2d": "Walker2d-v4",
    "swimmer": "Swimmer-v4",
    "reacher": "Reacher-v4",
    "pendulum": "Pendulum-v1",
    # Non-MuJoCo but useful classic control
    "cartpole": "CartPole-v1",
}

# Removed custom XML loader; we strictly rely on Gymnasium's built-in environments.


def make_env(name: str, render: bool = False):
    """Create a Gymnasium MuJoCo environment.

    Parameters
    ----------
    name : str
        Short key identifying the environment (e.g. ``"ant"``).
    render : bool, optional
        If ``True`` the environment will be created in human-rendering mode.

    Returns
    -------
    gym.Env
        Instantiated environment.
    """
    name = name.lower()

    gym_id = MUJOCO_GYM_MAP.get(name)
    if gym_id is None:
        # Attempt generic fallback (e.g. "Ant-v4" from "Ant")
        gym_id = f"{name.capitalize()}-v4"
        try:
            gym.spec(gym_id)
        except gym_error.Error as exc:
            raise ValueError(
                f"Unknown environment key '{name}'. Known keys: {list(MUJOCO_GYM_MAP.keys())}"
            ) from exc

    kwargs = {"render_mode": "human"} if render else {}
    return gym.make(gym_id, **kwargs) 