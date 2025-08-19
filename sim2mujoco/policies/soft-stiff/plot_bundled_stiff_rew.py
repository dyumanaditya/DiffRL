import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch  # ← added

def find_event_file(run_path):
    """
    Recursively search for a TensorBoard event file in the given run directory.
    Returns the earliest matching file path.
    """
    pattern = os.path.join(run_path, '**', '*tfevents*')
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_path}")
    files.sort(key=lambda f: os.path.getmtime(f))
    return files[0]

def find_run_dirs(root_path):
    """
    Return a list of *run directories* under root_path that contain tfevents files.
    Each directory is treated as one seeded run.
    """
    pattern = os.path.join(root_path, '**', '*tfevents*')
    files = glob.glob(pattern, recursive=True)
    run_dirs = sorted(set(os.path.dirname(f) for f in files))
    if not run_dirs:
        raise FileNotFoundError(f"No TensorBoard event files found under {root_path}")
    return run_dirs

def load_rewards(run_dir, tag='rewards'):
    """
    Load scalar values for `tag` from all TensorBoard event files inside run_dir.
    Returns a DataFrame indexed by `step` with column `reward`.
    """
    ea = event_accumulator.EventAccumulator(run_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    scalars = ea.Tags().get('scalars', [])
    if tag not in scalars:
        raise KeyError(f"Tag '{tag}' not found in {run_dir}")
    events = ea.Scalars(tag)
    df = pd.DataFrame({
        'step': [e.step for e in events],
        'reward': [e.value for e in events]
    }).drop_duplicates(subset='step').set_index('step').sort_index()
    return df

def aggregate_runs_mean_std(run_dirs, tag, num_points=400):
    """
    Given a list of run directories, load each run's rewards and compute
    mean and std on a common step grid using linear interpolation.
    """
    series = []
    for rd in run_dirs:
        try:
            df = load_rewards(rd, tag)
            series.append((df.index.values.astype(float), df['reward'].values.astype(float)))
        except Exception as e:
            print(f"[!] Skipping {rd}: {e}")

    if not series:
        raise RuntimeError("No valid runs to aggregate.")

    # Use overlapping region across all runs
    start = max(s.min() for s, _ in series)
    end   = min(s.max() for s, _ in series)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise RuntimeError("Runs do not share an overlapping step range.")

    steps_common = np.linspace(start, end, num_points)
    stacked = np.stack([np.interp(steps_common, s, v) for s, v in series], axis=0)
    mean = stacked.mean(axis=0)
    std  = stacked.std(axis=0)
    return steps_common, mean, std

def plot_three_runs(paths, tag, smooth_window, output_path):
    """
    Plot mean ± std as a shaded area for THREE experiment folders.
    Each folder may contain multiple seeded runs (subfolders with tfevents).
    """
    sns.set()
    plt.figure(figsize=(10, 6))

    labels = ["SHAC Low Stiffness", "SHAC High Stiffness + Bundle", "SHAC High Stiffness"]

    ax = plt.gca()
    for i, (path, label) in enumerate(zip(paths, labels)):
        try:
            run_dirs = find_run_dirs(path)
        except Exception as e:
            print(f"[!] {path}: {e}")
            continue

        try:
            steps, mean, std = aggregate_runs_mean_std(run_dirs, tag, num_points=400)
        except Exception as e:
            print(f"[!] {path}: {e}")
            continue

        # Optional smoothing on mean/std (keeps same length)
        if smooth_window > 1:
            mean = pd.Series(mean).rolling(window=smooth_window, min_periods=1, center=True).mean().values
            std  = pd.Series(std).rolling(window=smooth_window, min_periods=1, center=True).mean().values

        # Choose colors per series
        if i == 2:
            color = 'red'
        elif i == 1:
            color = 'C2'
        else:
            color = None  # let mpl pick

        # Plot mean line
        line = ax.plot(steps, mean, label=label, color=color)[0]
        line_color = line.get_color()

        # Shaded area: mean ± std
        ax.fill_between(steps, mean - std, mean + std, alpha=0.25, color=line_color)

    # format x-axis ticks in millions
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x)}"))
    plt.xlabel('Step', fontsize=12)
    plt.ylabel("Reward", fontsize=12)

    # --- Legend as color blocks below the plot ---
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_colors = []
    for h in legend_handles:
        # Line2D: get_color(); fallback to facecolor if needed
        c = getattr(h, "get_color", None)
        if callable(c):
            legend_colors.append(h.get_color())
        else:
            # e.g., PolyCollection
            fc = getattr(h, "get_facecolor", None)
            legend_colors.append(fc()[0] if callable(fc) else "gray")

    patches = [Patch(facecolor=c, edgecolor=c, label=l, linewidth=0)
               for c, l in zip(legend_colors, legend_labels)]

    leg = ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(patches),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        handlelength=2.2,
        handletextpad=0.5,
        columnspacing=1.0,
        fontsize=12
    )
    # Optional rounded legend box:
    # leg.get_frame().set_boxstyle("round,pad=0.3")

    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')  # ensure the below-plot legend is included
    plt.show()
    print(f"Saved plot with three runs to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot mean±std reward curves with shading for three experiment folders (each may contain multiple seeded runs).')
    parser.add_argument('run_dirs', nargs=3, type=str,
                        help='Paths to three experiment folders (each containing one or more seeded runs)')
    parser.add_argument('-t', '--tag', type=str, default='rewards',
                        help='TensorBoard scalar tag to plot')
    parser.add_argument('-w', '--smooth-window', type=int, default=1,
                        help='Window size for running average (1 = no smoothing)')
    parser.add_argument('-o', '--output', type=str, default='three_runs_comparison.png',
                        help='Path for output image')
    args = parser.parse_args()

    for p in args.run_dirs:
        if not os.path.isdir(p):
            parser.error(f"Run directory not found: {p}")

    plot_three_runs(args.run_dirs, args.tag, args.smooth_window, args.output)

if __name__ == '__main__':
    main()

