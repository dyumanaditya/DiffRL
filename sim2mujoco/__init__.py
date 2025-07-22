from importlib import import_module

__all__ = [
    "make_env",
    "MUJOCO_GYM_MAP",
    "load_policy",
    "evaluate",
]

# Lazy import submodules when accessed – keeps import times low.

def __getattr__(name):
    if name in ("make_env", "MUJOCO_GYM_MAP"):
        mod = import_module("sim2mujoco.envs")
        return getattr(mod, name)
    elif name in ("load_policy",):
        mod = import_module("sim2mujoco.policy")
        return getattr(mod, name)
    elif name in ("evaluate",):
        mod = import_module("sim2mujoco.runner")
        return getattr(mod, name)
    raise AttributeError(f"module 'sim2mujoco' has no attribute {name}") 