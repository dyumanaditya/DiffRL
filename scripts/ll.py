# loss_landscape.py
import os, sys, shutil, yaml, math, time
import numpy as np
from datetime import datetime

from shac.utils import hydra_utils
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---- existing project imports (kept identical) ----
from gym import wrappers
from rl_games.common import env_configurations, vecenv
from shac.utils.common import seeding, print_warning

try:
    from svg.train import Workspace
except Exception:
    print_warning("SVG not installed")

# Expose sim2mujoco envs as in your original file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ----------------------------
# Env registry (copied style)
# ----------------------------
def register_envs(env_config):
    def create_dflex_env(**kwargs):
        env = instantiate(env_config.config, no_grad=True)
        frames = kwargs.pop("frames", 1)
        if frames > 1:
            env = wrappers.FrameStack(env, frames, False)
        return env

    def create_warp_env(**kwargs):
        env = instantiate(env_config.config, no_grad=True)
        frames = kwargs.pop("frames", 1)
        if frames > 1:
            env = wrappers.FrameStack(env, frames, False)
        return env

    vecenv.register(
        "DFLEX",
        lambda config_name, num_actors, **kwargs: None,  # not used here
    )
    env_configurations.register(
        "dflex",
        {"env_creator": lambda **kwargs: create_dflex_env(**kwargs), "vecenv_type": "DFLEX"},
    )

    vecenv.register(
        "WARP",
        lambda config_name, num_actors, **kwargs: None,  # not used here
    )
    env_configurations.register(
        "warp",
        {"env_creator": lambda **kwargs: create_warp_env(**kwargs), "vecenv_type": "WARP"},
    )


# -------------------------------------
# Utilities: policy access + directions
# -------------------------------------
def _device_of(module: nn.Module):
    for p in module.parameters():
        return p.device
    return torch.device("cpu")


def find_policy_module(algo) -> nn.Module:
    """
    Best-effort grab for the policy/actor nn.Module inside various algs (SHAC/PPO/etc).
    Adjust if your project keeps it elsewhere.
    """
    candidate_attrs = [
        "actor", "policy", "pi", "net", "model", "actor_critic", "agent",
        "module", "policy_net", "actor_net"
    ]
    # Direct attributes first
    for name in candidate_attrs:
        mod = getattr(algo, name, None)
        if isinstance(mod, nn.Module):
            return mod
        # rl_games-style nesting
        if hasattr(algo, name) and hasattr(getattr(algo, name), "__dict__"):
            # search 1 level deeper
            for k, v in getattr(algo, name).__dict__.items():
                if isinstance(v, nn.Module):
                    return v

    # Fallback: first nn.Module field anywhere on algo
    for k, v in algo.__dict__.items():
        if isinstance(v, nn.Module):
            return v

    raise RuntimeError("Could not locate policy nn.Module on the algorithm. "
                       "Please set `policy = your_algo.<path_to_policy_module>` and use that.")


def params_list(policy: nn.Module, include_bias=True):
    named = list(policy.named_parameters())
    if not include_bias:
        named = [(n, p) for n, p in named if not n.endswith("bias")]
    # filter frozen
    named = [(n, p) for n, p in named if p.requires_grad]
    return named


def filter_normalize_direction(W: torch.Tensor, D: torch.Tensor, mode: str = "filter") -> torch.Tensor:
    """
    Scale random direction D so each filter-row has same norm as corresponding W filter-row.
    If `mode="layer"`, match full-tensor Frobenius norm instead.
    """
    if mode == "layer" or W.ndim < 2:
        w_norm = torch.linalg.norm(W) + 1e-12
        d_norm = torch.linalg.norm(D) + 1e-12
        return D * (w_norm / d_norm)

    # per-filter
    out = W.shape[0]
    Wf = W.reshape(out, -1)
    Df = D.reshape(out, -1)
    w_norms = torch.linalg.norm(Wf, dim=1) + 1e-12
    d_norms = torch.linalg.norm(Df, dim=1) + 1e-12
    scale = (w_norms / d_norms).unsqueeze(1)
    Dn = (Df * scale).reshape_as(D)
    return Dn


def sample_filter_normalized_directions(named_params, rng=None, mode="filter"):
    """
    Samples two random directions with per-filter (or layer) normalization.
    Compatible with older PyTorch versions that don't support the `generator`
    kwarg on torch.randn_like.
    """
    d1 = []
    d2 = []
    with torch.no_grad():
        for _, p in named_params:
            # Try the modern API first; fall back to version-safe path.
            try:
                if rng is not None:
                    r1 = torch.randn_like(p, generator=rng)
                    r2 = torch.randn_like(p, generator=rng)
                else:
                    r1 = torch.randn_like(p)
                    r2 = torch.randn_like(p)
            except TypeError:
                # Older PyTorch: ignore `generator` and rely on torch.manual_seed(...)
                r1 = torch.randn_like(p)
                r2 = torch.randn_like(p)

            d1.append(filter_normalize_direction(p, r1, mode))
            d2.append(filter_normalize_direction(p, r2, mode))
    return d1, d2


class ParamSwapper:
    """Context manager to swap parameters to (p0 + a*d1 + b*d2) and restore afterwards."""
    def __init__(self, named_params, base_params, d1, d2):
        self._params = [p for _, p in named_params]
        self._base = [b.detach().clone() for b in base_params]
        self._d1 = d1
        self._d2 = d2

    def set(self, alpha: float, beta: float):
        with torch.no_grad():
            for p, p0, u, v in zip(self._params, self._base, self._d1, self._d2):
                p.copy_(p0 + alpha * u + beta * v)

    def restore(self):
        with torch.no_grad():
            for p, p0 in zip(self._params, self._base):
                p.copy_(p0)


# ----------------------------------------
# Rollout / evaluation using current policy
# ----------------------------------------
@torch.no_grad()
def _policy_action_any(algo, policy, obs):
    """
    Try common ways to get an action. Customize if your code differs.
    Expects `obs` as a torch tensor [batch, ...] or numpy array; returns numpy action.
    """
    # algo methods first
    for fn in ["act", "select_action", "get_action", "inference", "inference_fn"]:
        f = getattr(algo, fn, None)
        if callable(f):
            out = f(obs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out.detach().cpu().numpy() if torch.is_tensor(out) else np.asarray(out)

    # policy methods
    for fn in ["act", "forward", "__call__"]:
        f = getattr(policy, fn, None)
        if callable(f):
            out = f(obs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out.detach().cpu().numpy() if torch.is_tensor(out) else np.asarray(out)

    raise RuntimeError("Could not infer how to compute an action from the policy. "
                       "Implement `_policy_action_any` for your policy.")


@torch.no_grad()
def evaluate_returns(env, algo, policy, episodes: int, device):
    """
    Preferred: if the algorithm exposes `evaluate_policy` (SHAC does),
    call it and convert the reported loss back to return.
    Fallback: do a manual single-env loop with torch tensors.
    """
    # 1) SHAC path: use the built-in evaluator (it handles tanh + tensor I/O)
    eval_fn = getattr(algo, "evaluate_policy", None)
    if callable(eval_fn):
        mean_policy_loss, _, _ = eval_fn(num_games=int(episodes), deterministic=True)
        mean_return = -float(mean_policy_loss)
        return mean_return, 0.0  # std not available from this API

    # 2) Generic fallback (torch-only, applies tanh if needed)
    returns = []

    # Prefer the algo's own env if present
    env_obj = getattr(algo, "env", env)

    for _ in range(int(episodes)):
        # Ensure torch obs on the right device
        obs = env_obj.reset()
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
        else:
            obs = obs.to(device)

        # roll one episode
        ep_ret = 0.0
        done = False
        while not done:
            # Try common policy call patterns to get a torch action
            act = None
            for fn_name in ["act", "forward", "__call__"]:
                fn = getattr(policy, fn_name, None)
                if callable(fn):
                    out = fn(obs if obs.dim() == 2 else obs.unsqueeze(0))
                    act = out[0] if isinstance(out, (list, tuple)) else out
                    break
            if act is None:
                raise RuntimeError("Could not compute action from policy; implement a policy call here.")

            # Heuristic: squash if magnitude suggests unbounded logits
            if torch.max(torch.abs(act)).item() > 1.05:
                act = torch.tanh(act)

            # Ensure shape [num_envs, act_dim]
            if act.dim() == 1:
                act = act.unsqueeze(0)

            # Step env with **torch** actions
            next_obs, rew, term_or_done, info = env_obj.step(act)

            # Handle Gym/Gymnasium variants and torch tensors
            if isinstance(term_or_done, tuple) and len(term_or_done) == 2:
                terminated, truncated = term_or_done
                done = bool(terminated) or bool(truncated)
            else:
                done = bool(term_or_done.item() if torch.is_tensor(term_or_done) else term_or_done)

            # Convert reward to float
            r = rew
            if torch.is_tensor(r):
                # vector env -> take first element if present
                r = r[0] if r.dim() > 0 else r
                r = float(r.item())
            elif isinstance(r, (list, np.ndarray)):
                r = float(r[0])
            else:
                r = float(r)
            ep_ret += r

            # Prepare next obs
            obs = next_obs
            if not torch.is_tensor(obs):
                obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
            else:
                obs = obs.to(device)

        returns.append(ep_ret)

    mean = float(np.mean(returns))
    std = float(np.std(returns)) if len(returns) > 1 else 0.0
    return mean, std


# -----------------------
# Plotting + save helpers
# -----------------------
def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"[saved] {path}")

def plot_contour(X, Y, Z, out_path, title="Loss landscape (−return)"):
    plt.figure(figsize=(7, 6))
    cs = plt.contourf(X, Y, Z, levels=30)
    plt.colorbar(cs, label="Loss (lower is better)")
    plt.scatter([0.0], [0.0], marker="x", s=80, linewidths=2, label="checkpoint")
    plt.xlabel("alpha (dir1)")
    plt.ylabel("beta (dir2)")
    plt.legend(loc="upper right")
    plt.title(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")

def plot_surface(X, Y, Z, out_path, title="3D Loss surface (−return)"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    ax.set_xlabel("alpha (dir1)")
    ax.set_ylabel("beta (dir2)")
    ax.set_zlabel("Loss")
    ax.set_title(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")


# -----------------------
# Main Hydra entry point
# -----------------------
@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    """
    Run a 2-D loss landscape sweep by perturbing a loaded policy checkpoint along two
    filter-normalized random directions and evaluating episodic return.
    """
    # Defaults for landscape-related args (can be overridden with `landscape.*`)
    default_landscape = {
        "grid": 21,        # grid size per axis
        "span": 0.005,       # +/- range of alphas/betas
        "episodes": 1,     # eval episodes per grid point
        "seed": 0,         # direction RNG seed
        "normalize": "filter",  # "filter" or "layer"
        "include_bias": True,
        "grid_from": None,          # path to an existing loss_landscape_grid.npz (reuses alphas/betas)
        "directions_from": None,    # path to directions.pt (reuses d1/d2)
        "save_directions": True,    # save directions for this run
        "directions_filename": "directions.pt",  # filename under run dir
    }
    if "landscape" not in cfg:
        with open_dict(cfg):
            cfg.landscape = default_landscape
    else:
        for k, v in default_landscape.items():
            if k not in cfg.landscape:
                with open_dict(cfg):
                    cfg.landscape[k] = v

    # Logging dir similar to your train()
    main_logdir = cfg.general.logdir
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S.%f")
    root_dir = os.path.join("outputs", main_logdir, date_str, time_str, "loss_landscape")
    os.makedirs(root_dir, exist_ok=True)
    print(f"Writing outputs to: {root_dir}")

    # Copy Hydra outputs next to our logs (like your train.py)
    hydra_output_dir = HydraConfig.get().run.dir
    if hydra_output_dir and os.path.exists(hydra_output_dir):
        target_dir = os.path.dirname(root_dir)
        for item in os.listdir(hydra_output_dir):
            src = os.path.join(hydra_output_dir, item)
            dst = os.path.join(target_dir, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)

    # Seeding (PyTorch + numpy + envs via your util)
    seed = int(cfg.landscape.seed)
    seeding(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = torch.Generator().manual_seed(seed)

    # Build env in no-grad eval mode; force single env for stability
    with open_dict(cfg):
        cfg.general.train = False
        cfg.env.config.no_grad = True
        if "num_envs" in cfg.env.config and cfg.env.config.num_envs != 1:
            cfg.env.config.num_envs = 1

    # If your config supports sim2mujoco, just use the normal env
    register_envs(cfg.env if "mujoco" not in cfg.env else cfg.env.mujoco)

    env = instantiate(cfg.env.config, no_grad=True)

    # Instantiate algorithm and load checkpoint
    algo = instantiate(cfg.alg, env_config=cfg.env.config, logdir=root_dir)
    if cfg.general.checkpoint:
        print(f"Loading checkpoint: {cfg.general.checkpoint}")
        algo.load(cfg.general.checkpoint)
    else:
        raise RuntimeError("Please provide a checkpoint via `general.checkpoint=/path/to/ckpt`")

    # Find policy module + param list
    policy = find_policy_module(algo)
    device = _device_of(policy)
    named = params_list(policy, include_bias=bool(cfg.landscape.include_bias))
    base = [p.detach().clone() for _, p in named]

    # -------------------------
    # Grid: load or create
    # -------------------------
    use_grid = cfg.landscape.get("grid_from", None)
    if use_grid:
        data = np.load(use_grid)
        alphas = data["alphas"]
        betas = data["betas"]
        print(f"Loaded α/β grid from: {use_grid}  (len(alphas)={len(alphas)}, len(betas)={len(betas)})")
    else:
        G = int(cfg.landscape.grid)
        span = float(cfg.landscape.span)
        alphas = np.linspace(-span, span, G, dtype=np.float32)
        betas  = np.linspace(-span, span, G, dtype=np.float32)

    AA, BB = np.meshgrid(alphas, betas, indexing="xy")
    Z = np.zeros_like(AA, dtype=np.float32)

    # -------------------------
    # Directions: load or sample
    # -------------------------
    directions_from = cfg.landscape.get("directions_from", None)
    if directions_from:
        pkg = torch.load(directions_from, map_location=device)
        names_here = [n for n, _ in named]
        shapes_here = [tuple(p.shape) for _, p in named]
        assert names_here == pkg["names"], "Param names/order differ from saved directions."
        assert shapes_here == [tuple(s) for s in pkg["shapes"]], "Param shapes differ from saved directions."
        d1 = [t.to(device) for t in pkg["d1"]]
        d2 = [t.to(device) for t in pkg["d2"]]
        print(f"Loaded directions from: {directions_from}")
        print(f"(normalize was '{pkg.get('normalize','filter')}', include_bias={pkg.get('include_bias', True)}, seed={pkg.get('seed','?')})")
    else:
        d1, d2 = sample_filter_normalized_directions(named, rng, mode=str(cfg.landscape.normalize))
        print("Sampled new directions (filter/layer-normalized).")
        if bool(cfg.landscape.save_directions):
            directions_path = os.path.join(root_dir, str(cfg.landscape.directions_filename))
            torch.save({
                "names": [n for n, _ in named],
                "shapes": [tuple(p.shape) for _, p in named],
                "d1": [t.detach().cpu() for t in d1],
                "d2": [t.detach().cpu() for t in d2],
                "normalize": str(cfg.landscape.normalize),
                "include_bias": bool(cfg.landscape.include_bias),
                "seed": int(cfg.landscape.seed),
            }, directions_path)
            print(f"[saved] directions -> {directions_path}")

    swapper = ParamSwapper(named, base, d1, d2)

    # Evaluate baseline
    swapper.set(0.0, 0.0)
    mean_ret0, std_ret0 = evaluate_returns(env, algo, policy, int(cfg.landscape.episodes), device)
    base_loss = -mean_ret0
    print(f"Baseline return: {mean_ret0:.4f} ± {std_ret0:.4f}")

    # Sweep
    Gx, Gy = len(alphas), len(betas)
    total = Gx * Gy
    t0 = time.time()
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            swapper.set(float(a), float(b))
            mean_ret, std_ret = evaluate_returns(env, algo, policy, int(cfg.landscape.episodes), device)
            loss = -mean_ret
            Z[j, i] = loss  # note [beta, alpha] to match meshgrid XY
            done = i * Gy + j + 1
            if done % max(1, total // 20) == 0:
                dt = time.time() - t0
                print(f"[{done}/{total}] a={a:+.3f} b={b:+.3f} loss={loss:+.4f} (elapsed {dt:.1f}s)")

    # Restore original params
    swapper.restore()

    # Save data
    npz_path = os.path.join(root_dir, "loss_landscape_grid.npz")
    save_npz(
        npz_path,
        alphas=alphas,
        betas=betas,
        Z=Z,
        baseline_loss=np.asarray([base_loss], dtype=np.float32),
        baseline_return=np.asarray([mean_ret0], dtype=np.float32),
        normalize=np.asarray([0 if cfg.landscape.normalize == "layer" else 1], dtype=np.int32),
        include_bias=np.asarray([1 if cfg.landscape.include_bias else 0], dtype=np.int32),
        seed=np.asarray([seed], dtype=np.int32),
    )

    # Plots
    contour_path = os.path.join(root_dir, "loss_landscape_contour.png")
    surface_path = os.path.join(root_dir, "loss_landscape_surface.png")
    plot_contour(AA, BB, Z, contour_path, title="Loss landscape (−episodic return)")
    plot_surface(AA, BB, Z, surface_path, title="3D Loss surface (−episodic return)")

    print("\nDone.")
    print(f"Data:    {npz_path}")
    if not directions_from and bool(cfg.landscape.save_directions):
        print(f"Dirs:    {os.path.join(root_dir, str(cfg.landscape.directions_filename))}")
    print(f"2D plot: {contour_path}")
    print(f"3D plot: {surface_path}")


if __name__ == "__main__":
    main()
