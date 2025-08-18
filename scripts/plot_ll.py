# plot_loss_landscape.py (modified per request)
import os
import argparse
import numpy as np
import matplotlib
from glob import glob
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---- parse args BEFORE importing pyplot so we can set backend ----
def build_argparser():
    ap = argparse.ArgumentParser(description="Plot saved loss landscape grids (.npz).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in", dest="inp", type=str, help="Path to a single loss_landscape_grid.npz")
    g.add_argument("--glob", dest="pattern", type=str, help='Glob pattern, e.g. "outputs/**/loss_landscape_grid.npz"')
    g.add_argument("--compare", dest="compare", nargs=2, type=str, help="Compare two runs: --compare run1.npz run2.npz")

    ap.add_argument("--outdir", type=str, default=None, help="Directory to write plots. Default: alongside each npz")
    ap.add_argument("--vmin", type=float, default=None, help="Fix min color scale (use with --vmax for comparisons)")
    ap.add_argument("--vmax", type=float, default=None, help="Fix max color scale")
    ap.add_argument("--levels", type=int, default=30, help="Number of contour levels")
    ap.add_argument("--prefix", type=str, default=None, help="Optional prefix for output filenames")

    # interactive flags
    ap.add_argument("--show", action="store_true", help="Show figures before saving (requires interactive backend)")
    ap.add_argument("--backend", type=str, default=None,
                    help="Matplotlib backend to use when --show (e.g., TkAgg, QtAgg, Qt5Agg, MacOSX, WebAgg).")
    ap.add_argument("--block", dest="block", action="store_true", help="Block on plt.show() (default).")
    ap.add_argument("--no-block", dest="block", action="store_false", help="Do not block on plt.show().")
    ap.set_defaults(block=True)

    # loss scaling option
    ap.add_argument("--loss-scale", type=str, default="linear",
                    choices=["linear", "symlog10"],
                    help="Color/loss scaling for contour & surface. Use 'symlog10' for symmetric log10 scale.")
    return ap

ap = build_argparser()
args = ap.parse_args()

# Decide backend
if args.show:
    if args.backend:
        try:
            matplotlib.use(args.backend, force=True)
        except Exception as e:
            print(f"[warn] Failed to set backend '{args.backend}': {e}. Falling back to auto.")
            args.backend = None
    if args.backend is None:
        for cand in ("QtAgg", "Qt5Agg", "TkAgg", "MacOSX", "WebAgg"):
            try:
                matplotlib.use(cand, force=True)
                args.backend = cand
                break
            except Exception:
                continue
    if args.backend is None:
        print("[warn] No interactive backend available. Falling back to Agg (non-interactive).")
        matplotlib.use("Agg", force=True)
        args.show = False
else:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------- global style: font sizes ----------
plt.rcParams.update({
    "axes.labelsize": 24,     # x/y/z label size
    "legend.fontsize": 24,    # legend font size
    "xtick.labelsize": 20,    # tick font sizes
    "ytick.labelsize": 20,
})

def load_grid(npz_path):
    data = np.load(npz_path)
    alphas = data["alphas"]
    betas = data["betas"]
    Z = data["Z"]
    baseline_return = float(data["baseline_return"][0]) if "baseline_return" in data else None
    baseline_loss = float(data["baseline_loss"][0]) if "baseline_loss" in data else None
    meta = {
        "normalize": "filter" if int(data.get("normalize", np.array([1]))[0]) == 1 else "layer",
        "include_bias": bool(int(data.get("include_bias", np.array([1]))[0])),
        "seed": int(data.get("seed", np.array([0]))[0]),
    }
    X, Y = np.meshgrid(alphas, betas, indexing="xy")
    return X, Y, Z, baseline_return, baseline_loss, meta

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def finalize_figure(fig, out_path, show=False, block=True, save=True, save_after_show=True):
    # Avoid tight_layout for 3D axes to prevent label/tick overlap
    try:
        has_3d = any(getattr(ax, "name", "") == "3d" for ax in fig.axes)
        if not has_3d:
            fig.tight_layout()
        fig.canvas.draw()
    except Exception:
        pass

    if show:
        try:
            plt.show(block=block)
        except Exception as e:
            print(f"[warn] plt.show() failed: {e}")

    if save:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        print(f"[saved] {out_path}")

    plt.close(fig)

def plot_contour(X, Y, Z, out_path, vmin=None, vmax=None, levels=30, show=False, block=True):
    from matplotlib import colors, cm
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    vmin_eff = float(np.nanmin(Z)) if vmin is None else vmin
    vmax_eff = float(np.nanmax(Z)) if vmax is None else vmax

    cmap = cm.get_cmap("RdYlBu_r")
    if getattr(args, "loss_scale", "linear") == "symlog10":
        norm = colors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=vmin_eff, vmax=vmax_eff, base=10)
    else:
        norm = colors.Normalize(vmin=vmin_eff, vmax=vmax_eff, clip=True)

    cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)

    # Colorbar with more gap to the axes, and label farther from the colormap
    cbar = fig.colorbar(cs, ax=ax, pad=0.10)
    cbar.ax.set_ylabel("Loss", fontsize=24, labelpad=18)
    cbar.ax.tick_params(labelsize=20)

    ax.scatter([0.0], [0.0], marker="x", s=80, linewidths=2, label="checkpoint")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.tick_params(axis="both", labelsize=20)
    ax.legend(loc="upper right", fontsize=24)

    finalize_figure(fig, out_path, show=show, block=block, save=True, save_after_show=True)

def plot_heatmap(X, Y, Z, out_path, vmin=None, vmax=None, show=False, block=True):
    from matplotlib import cm
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        Z, origin="lower",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        aspect="auto", vmin=vmin, vmax=vmax,
    )

    # Colorbar with more gap to the axes, and label farther from the colormap
    cbar = fig.colorbar(im, ax=ax, pad=0.10)
    cbar.ax.set_ylabel("Loss", fontsize=24, labelpad=18)
    cbar.ax.tick_params(labelsize=20)

    ax.scatter([0.0], [0.0], marker="x", s=80, linewidths=2)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.tick_params(axis="both", labelsize=20)

    finalize_figure(fig, out_path, show=show, block=block, save=True, save_after_show=True)

def plot_surface(X, Y, Z, out_path, show=False, block=True):
    # Colormapped 3D surface with colorbar; uses --loss-scale for normalization and RdYlBu_r colormap.
    from matplotlib import cm, colors

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Color normalization (prefer CLI-specified vmin/vmax if set)
    vmin = float(np.nanmin(Z)) if getattr(args, "vmin", None) is None else args.vmin
    vmax = float(np.nanmax(Z)) if getattr(args, "vmax", None) is None else args.vmax

    if getattr(args, "loss_scale", "linear") == "symlog10":
        norm = colors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=vmin, vmax=vmax, base=10)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    cmap = cm.get_cmap("RdYlBu_r")

    # Facecolors derived from Z; shade=False keeps the colormap faithful
    facecolors = cmap(norm(Z))
    ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)

    # Colorbar keyed to Z, with a bit more padding
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(Z)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.22)
    cbar.ax.set_ylabel("Loss", fontsize=24, labelpad=16)
    cbar.ax.tick_params(labelsize=18)

    # Axis limits
    ax.set_zlim(vmin, vmax)

    # Axis labels: turn off rotation for z and add generous padding
    ax.set_xlabel(r"$\alpha$", labelpad=22)
    ax.set_ylabel(r"$\beta$", labelpad=22)
    ax.zaxis.set_rotate_label(False)           # don't auto-rotate
    ax.set_zlabel("Loss", rotation=90, labelpad=40)

    # Tick label spacing and size (shrink to avoid overlap)
    ax.tick_params(axis="x", pad=12, labelsize=18)
    ax.tick_params(axis="y", pad=12, labelsize=18)
    ax.tick_params(axis="z", pad=14, labelsize=18)

    # A bit of margins so labels have room
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    finalize_figure(fig, out_path, show=show, block=block, save=True, save_after_show=True)

def plot_slices(X, Y, Z, out_path, show=False, block=True):
    # nearest indices to alpha=0, beta=0
    i0 = int(np.argmin(np.abs(X[0, :])))
    j0 = int(np.argmin(np.abs(Y[:, 0])))
    alpha_line = Z[j0, :]  # beta ~ 0, vary alpha
    beta_line  = Z[:, i0]  # alpha ~ 0, vary beta

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(X[0, :], alpha_line, label=r"$\beta \approx 0$ (vary $\alpha$)")
    ax.plot(Y[:, 0], beta_line,  label=r"$\alpha \approx 0$ (vary $\beta$)")
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\alpha$ or $\beta$")
    ax.set_ylabel("Loss")
    ax.tick_params(axis="both", labelsize=20)
    ax.legend(fontsize=24)

    finalize_figure(fig, out_path, show=show, block=block, save=True, save_after_show=True)

def plot_comparison_contour(X1, Y1, Z1, X2, Y2, Z2, out_path, vmin=None, vmax=None, levels=30, show=False, block=True, label1="Run 1", label2="Run 2"):
    from matplotlib import colors, cm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Determine global min/max for consistent scaling
    vmin_eff = float(min(np.nanmin(Z1), np.nanmin(Z2))) if vmin is None else vmin
    vmax_eff = float(max(np.nanmax(Z1), np.nanmax(Z2))) if vmax is None else vmax

    cmap = cm.get_cmap("RdYlBu_r")
    if getattr(args, "loss_scale", "linear") == "symlog10":
        norm = colors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=vmin_eff, vmax=vmax_eff, base=10)
    else:
        norm = colors.Normalize(vmin=vmin_eff, vmax=vmax_eff, clip=True)

    # First plot
    cs1 = ax1.contourf(X1, Y1, Z1, levels=levels, cmap=cmap, norm=norm)
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"$\beta$")
    ax1.tick_params(axis="both", labelsize=20)
    ax1.set_title(label1, fontsize=24)

    # Second plot
    cs2 = ax2.contourf(X2, Y2, Z2, levels=levels, cmap=cmap, norm=norm)
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel(r"$\beta$")
    ax2.tick_params(axis="both", labelsize=20)
    ax2.set_title(label2, fontsize=24)

    # Create colorbar with explicit positioning to avoid overlap
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    
    # Create a mappable object that covers both datasets for correct colorbar range
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.concatenate([Z1.flatten(), Z2.flatten()]))
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.set_ylabel("Loss", fontsize=24, labelpad=18)
    cbar.ax.tick_params(labelsize=20)

    # Use tight_layout to prevent overlap
    plt.tight_layout()

    finalize_figure(fig, out_path, show=show, block=block, save=True, save_after_show=True)

def plot_comparison_surface(X1, Y1, Z1, X2, Y2, Z2, out_path, show=False, block=True, label1="Run 1", label2="Run 2"):
    from matplotlib import cm, colors

    fig = plt.figure(figsize=(20, 7))
    
    # Determine global min/max for consistent scaling
    vmin = float(min(np.nanmin(Z1), np.nanmin(Z2))) if getattr(args, "vmin", None) is None else args.vmin
    vmax = float(max(np.nanmax(Z1), np.nanmax(Z2))) if getattr(args, "vmax", None) is None else args.vmax

    if getattr(args, "loss_scale", "linear") == "symlog10":
        norm = colors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=vmin, vmax=vmax, base=10)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    cmap = cm.get_cmap("RdYlBu_r")

    # First 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    facecolors1 = cmap(norm(Z1))
    ax1.plot_surface(X1, Y1, Z1, facecolors=facecolors1, rstride=1, cstride=1,
                     linewidth=0, antialiased=True, shade=False)
    ax1.set_xlabel(r"$\alpha$", labelpad=22)
    ax1.set_ylabel(r"$\beta$", labelpad=22)
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Loss", rotation=90, labelpad=40)
    ax1.tick_params(axis="x", pad=12, labelsize=18)
    ax1.tick_params(axis="y", pad=12, labelsize=18)
    ax1.tick_params(axis="z", pad=14, labelsize=18)
    ax1.set_zlim(vmin, vmax)
    ax1.set_title(label1, fontsize=24)

    # Second 3D plot
    ax2 = fig.add_subplot(122, projection='3d')
    facecolors2 = cmap(norm(Z2))
    ax2.plot_surface(X2, Y2, Z2, facecolors=facecolors2, rstride=1, cstride=1,
                     linewidth=0, antialiased=True, shade=False)
    ax2.set_xlabel(r"$\alpha$", labelpad=22)
    ax2.set_ylabel(r"$\beta$", labelpad=22)
    ax2.zaxis.set_rotate_label(False)
    ax2.set_zlabel("Loss", rotation=90, labelpad=40)
    ax2.tick_params(axis="x", pad=12, labelsize=18)
    ax2.tick_params(axis="y", pad=12, labelsize=18)
    ax2.tick_params(axis="z", pad=14, labelsize=18)
    ax2.set_zlim(vmin, vmax)
    ax2.set_title(label2, fontsize=24)

    # Create colorbar with manual positioning for 3D plots
    # Position colorbar to the right of both plots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.concatenate([Z1.flatten(), Z2.flatten()]))
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.ax.set_ylabel("Loss", fontsize=24, labelpad=16)
    cbar.ax.tick_params(labelsize=18)

    finalize_figure(fig, out_path, show=show, block=block, save=True, save_after_show=True)

def make_comparison_plots(npz_path1, npz_path2, outdir=None, vmin=None, vmax=None, levels=30, prefix=None, show=False, block=True):
    X1, Y1, Z1, base_ret1, base_loss1, meta1 = load_grid(npz_path1)
    X2, Y2, Z2, base_ret2, base_loss2, meta2 = load_grid(npz_path2)

    if outdir is None:
        outdir = Path(npz_path1).parent
    else:
        outdir = Path(outdir)
    ensure_dir(outdir)

    # Generate labels from filenames
    label1 = Path(npz_path1).parent.name
    label2 = Path(npz_path2).parent.name
    
    tag = prefix or "comparison"

    # Plot comparisons
    plot_comparison_contour(X1, Y1, Z1, X2, Y2, Z2, 
                           outdir / f"{tag}_contour.png", vmin, vmax, levels, show=show, block=block,
                           label1=label1, label2=label2)
    plot_comparison_surface(X1, Y1, Z1, X2, Y2, Z2, 
                           outdir / f"{tag}_surface.png", show=show, block=block,
                           label1=label1, label2=label2)

def make_all_plots(npz_path, outdir=None, vmin=None, vmax=None, levels=30, prefix=None, show=False, block=True):
    X, Y, Z, base_ret, base_loss, meta = load_grid(npz_path)

    if outdir is None:
        outdir = Path(npz_path).parent
    else:
        outdir = Path(outdir)
    ensure_dir(outdir)

    base_name = Path(npz_path).stem  # e.g., loss_landscape_grid
    tag = prefix or f"{base_name}"

    # No titles; filenames only
    plot_contour(X, Y, Z, outdir / f"{tag}_contour.png", vmin, vmax, levels, show=show, block=block)
    plot_heatmap(X, Y, Z, outdir / f"{tag}_heatmap.png", vmin, vmax, show=show, block=block)
    plot_surface(X, Y, Z, outdir / f"{tag}_surface.png", show=show, block=block)
    plot_slices(X, Y, Z, outdir / f"{tag}_slices.png", show=show, block=block)

def main():
    if args.compare:
        # Handle comparison case
        npz_path1, npz_path2 = args.compare
        if not os.path.exists(npz_path1):
            raise SystemExit(f"First file not found: {npz_path1}")
        if not os.path.exists(npz_path2):
            raise SystemExit(f"Second file not found: {npz_path2}")
        
        print(f"Comparing two runs:")
        print(f"- {npz_path1}")
        print(f"- {npz_path2}")
        
        make_comparison_plots(
            npz_path1=npz_path1,
            npz_path2=npz_path2,
            outdir=args.outdir,
            vmin=args.vmin,
            vmax=args.vmax,
            levels=args.levels,
            prefix=args.prefix,
            show=args.show,
            block=args.block,
        )
        return

    # Handle single file or glob cases
    files = []
    if args.inp:
        files = [args.inp]
    elif args.pattern:
        files = sorted(glob(args.pattern, recursive=True))
    if not files:
        raise SystemExit("No .npz files found to plot.")

    print(f"Found {len(files)} file(s).")
    for f in files:
        print(f"- {f}")
        make_all_plots(
            npz_path=f,
            outdir=args.outdir,
            vmin=args.vmin,
            vmax=args.vmax,
            levels=args.levels,
            prefix=args.prefix,
            show=args.show,
            block=args.block,
        )

if __name__ == "__main__":
    main()
