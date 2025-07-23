import torch
from typing import Union, Optional
from pathlib import Path

# Try importing rl_games; if not present, functionality will be skipped
try:
    from rl_games.algos_torch import model_builder, players, torch_ext  # type: ignore
    _RLGAMES_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RLGAMES_AVAILABLE = False

__all__ = ["load_policy"]


# -----------------------------------------------------------------------------
# RL-games helper
# -----------------------------------------------------------------------------


def _build_rl_player(weights_path: str, cfg: Union[str, dict], device: str):
    """Build an rl_games player from weights + config (path or dict)."""
    if not _RLGAMES_AVAILABLE:
        raise RuntimeError("rl_games package not available; cannot load PPO checkpoint")

    import yaml, torch, tempfile

    if isinstance(cfg, str):
        cfg_dict = yaml.safe_load(open(cfg))
    else:
        cfg_dict = cfg

    # Extract network and algo info
    net_cfg = cfg_dict["params"]["network"]
    algo_name = cfg_dict["params"]["algo"]["name"]

    # 1) create network builder
    net_builder = model_builder.NetworkBuilder()
    network_builder_inst = net_builder.load(net_cfg)

    # 2) choose correct player wrapper
    if algo_name == "a2c_continuous":
        player_cls = players.PpoPlayerContinuous
    elif algo_name == "a2c_discrete":
        player_cls = players.PpoPlayerDiscrete
    elif algo_name == "sac":
        player_cls = players.SACPlayer
    else:
        raise RuntimeError(f"Unsupported algo {algo_name} in rl_games checkpoint")

    player = player_cls(cfg_dict["params"])  # type: ignore

    # load weights (same logic as BasePlayer.restore)
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    player.model.load_state_dict(state_dict, strict=False)

    player.is_rnn = False

    class _PlayerWrapper(torch.nn.Module):
        def __init__(self, pl, dev):
            super().__init__()
            self._pl = pl
            self._device = dev

        def forward(self, obs: torch.Tensor, deterministic: bool = False):  # type: ignore
            obs_np = obs.detach().cpu().numpy()[None, ...]  # add batch dim
            action = self._pl.get_action(obs_np, is_deterministic=deterministic)[0]
            return torch.as_tensor(action, device=self._device, dtype=torch.float32)

    return _PlayerWrapper(player, device)

# -----------------------------------------------------------------------------
# Compose rl_games config from scripts/cfg directory
# -----------------------------------------------------------------------------


def _compose_cfg(env_key: str, algo_key: str) -> dict:
    """Recreate merged rl_games config using YAMLs in scripts/cfg."""
    from omegaconf import OmegaConf
    import yaml as _yaml

    root = Path(__file__).resolve().parents[1] / "scripts" / "cfg"

    base = OmegaConf.load(root / "config.yaml")
    algo_file = root / "alg" / f"{algo_key}.yaml"
    env_file = root / "env" / f"{env_key}.yaml"
    if not algo_file.exists():
        raise FileNotFoundError(algo_file)
    if not env_file.exists():
        raise FileNotFoundError(env_file)

    algo_cfg = OmegaConf.load(algo_file)
    env_cfg = OmegaConf.load(env_file)

    merged = OmegaConf.merge(base, algo_cfg)
    merged["params"]["diff_env"] = env_cfg["config"]

    # Provide ${env.*} interpolation sources expected by rl_games YAMLs
    if "env" not in merged:
        merged["env"] = {}
    
    # Add env.name (used in ${env.name} interpolations)
    merged["env"]["name"] = env_key
    
    # Add env.<algo_key> block (e.g. env.ppo.actor_mlp)
    algo_block_key = algo_key  # e.g. "ppo" or "sac"
    if algo_block_key not in merged["env"]:
        if algo_block_key in env_cfg:
            merged["env"][algo_block_key] = env_cfg[algo_block_key]
        else:
            merged["env"][algo_block_key] = {}

    # Matching train.py: add 'name' field for env (used in interpolations)
    env_target = env_cfg["config"].get("_target_", "")
    merged["params"]["diff_env"]["name"] = env_target.split(".")[-1]

    return _yaml.safe_load(OmegaConf.to_yaml(merged, resolve=True))


def _extract_actor(checkpoint: Union[list, dict, torch.nn.Module]):
    """Return the actor network from various checkpoint formats.

    The SHAC algorithm saves a *list* ``[actor, critic, target_critic, ...]``.
    Users might also provide a plain *state dict* or a scripted module. We try to
    handle the most common cases gracefully.
    """
    if isinstance(checkpoint, (list, tuple)) and len(checkpoint) >= 1:
        return checkpoint[0]

    # If the checkpoint is a dict try typical keys
    if isinstance(checkpoint, dict):
        for key in ("actor", "policy", "model", "policy_state_dict"):
            if key in checkpoint:
                return checkpoint[key]

    # Fallback – maybe the checkpoint itself is the actor module
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint

    raise RuntimeError(
        "Could not infer actor network from the provided checkpoint file – please check its format."
    )


def load_policy(path: str, device: str = "cpu", cfg_path: Optional[str] = None, *, env_key: Optional[str] = None, algo_key: str = "ppo") -> torch.nn.Module:
    """Load an actor network saved with SHAC (or similar) and move to *device*."""
    import os

    # If this is a rl_games checkpoint (endswith .pth) and yaml exists, use player loader
    if path.endswith(".pth") and _RLGAMES_AVAILABLE:
        if cfg_path is None:
            # attempt to locate yaml in same dir
            base_dir = os.path.dirname(path)
            for fname in os.listdir(base_dir):
                if fname.endswith( (".yaml", ".yml") ):
                    cfg_path = os.path.join(base_dir, fname)
                    break
        if cfg_path and os.path.isfile(cfg_path):
            try:
                return _build_rl_player(path, cfg_path, device)
            except Exception as e:
                print(f"[sim2mujoco] rl_games load failed: {e}. Falling back.")
        elif env_key is not None:
            # try:
            cfg_dict = _compose_cfg(env_key, algo_key)
            return _build_rl_player(path, cfg_dict, device)
            # except Exception as e:
            #     print(f"[sim2mujoco] auto-compose cfg failed: {e}. Falling back.")

    # First try regular torch.load (supports Python pickled modules, list, dict, …)
    checkpoint = torch.load(path, map_location=device)

    try:
        actor = _extract_actor(checkpoint)
    except RuntimeError:
        # Fallback: maybe the file is a TorchScript module (torch.jit.save)
        try:
            actor = torch.jit.load(path, map_location=device)
        except Exception as e:
            raise RuntimeError(
                "Could not load an actor network from the given checkpoint. "
                "If you saved only the model's `state_dict`, you must recreate "
                "the network architecture and load the state dict yourself."
            ) from e

    # If we ended up with a plain state_dict, return it so the caller can
    # attempt to reconstruct a matching architecture.
    if isinstance(actor, dict):
        return actor  # type: ignore – caller must handle

    actor = actor.to(device)
    actor.eval()
    return actor


# -----------------------------------------------------------------------------
#  Fallback: reconstruct an MLP policy from a *state_dict* only
# -----------------------------------------------------------------------------


def build_mlp_from_state_dict(
    state_dict: dict, obs_dim: int, act_dim: int, device: str = "cpu"
) -> torch.nn.Module:
    """Given a parameter state-dict with linear layer names, rebuild an MLP.

    Assumes layers are named like ``fc0.weight``, ``fc0.bias``, ``fc1.weight``, …
    or generic ``lin1.weight`` etc.  Orders layers by appearance in the state-dict.
    """
    import re
    from operator import itemgetter

    # Collect numeric-indexed layers first
    pattern = re.compile(r"(.*?)(\d+)\.weight$")
    layers_info = []  # (index, in_dim, out_dim, weight_key, bias_key)
    used_keys = set()

    for k, v in state_dict.items():
        m = pattern.match(k)
        if m:
            idx = int(m.group(2))
            out_dim, in_dim = v.shape
            bias_key = k.replace("weight", "bias")
            layers_info.append((idx, in_dim, out_dim, k, bias_key))
            used_keys.add(k)
            used_keys.add(bias_key)

    if not layers_info:
        raise RuntimeError("Cannot infer hidden layers from state_dict keys.")

    layers_info.sort(key=itemgetter(0))

    # Search for an explicit output layer (any remaining weight with out_dim == act_dim)
    output_layer = None  # (in_dim, out_dim, w_key, b_key)
    for k, v in state_dict.items():
        if k in used_keys or not k.endswith("weight"):
            continue
        out_dim, in_dim = v.shape
        if out_dim == act_dim:
            b_key = k.replace("weight", "bias")
            output_layer = (in_dim, out_dim, k, b_key)
            used_keys.add(k)
            used_keys.add(b_key)
            break

    # Build module list
    import torch.nn as nn
    modules = []

    # First layer: enforce obs_dim input
    first_in = obs_dim
    first_out = layers_info[0][2]
    modules.append(nn.Linear(first_in, first_out))
    modules.append(nn.Tanh())

    # Hidden layers (skip first because added)
    for prev, curr in zip(layers_info, layers_info[1:]):
        in_d = prev[2]
        out_d = curr[2]
        modules.append(nn.Linear(in_d, out_d))
        modules.append(nn.Tanh())

    # Output layer
    if output_layer is not None:
        modules.append(nn.Linear(output_layer[0], act_dim))
    else:
        # Fall back: use last hidden out dim
        modules.append(nn.Linear(layers_info[-1][2], act_dim))

    policy = nn.Sequential(*modules).to(device)

    # Load what we can
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    return policy
