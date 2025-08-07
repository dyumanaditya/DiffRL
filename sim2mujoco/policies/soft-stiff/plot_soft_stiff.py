import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
from matplotlib.ticker import FuncFormatter


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


def load_run_rewards(run_path, tag='rewards'):
    """
    Load scalar values for `tag` from the TensorBoard event file in run_path.
    Returns a DataFrame indexed by `step` with column `reward`.
    """
    event_file = find_event_file(run_path)
    ea = event_accumulator.EventAccumulator(event_file, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    scalars = ea.Tags().get('scalars', [])
    if tag not in scalars:
        raise KeyError(f"Tag '{tag}' not found in {event_file}")
    events = ea.Scalars(tag)
    df = pd.DataFrame({
        'step': [e.step for e in events],
        'reward': [e.value for e in events]
    }).drop_duplicates(subset='step').set_index('step')
    return df


def gather_experiment(root_dir, tag='rewards'):
    """
    Load rewards for each experiment under root_dir.
    Returns dict {exp_name: [DataFrame, ...]}.
    """
    exps = {}
    for exp_name in sorted(os.listdir(root_dir)):
        exp_path = os.path.join(root_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue
        dfs = []
        for run in sorted(os.listdir(exp_path)):
            run_path = os.path.join(exp_path, run)
            if not os.path.isdir(run_path):
                continue
            try:
                dfs.append(load_run_rewards(run_path, tag))
            except Exception as e:
                print(f"Warning: skipping {run_path}: {e}")
        if dfs:
            exps[exp_name] = dfs
        else:
            print(f"Warning: no valid runs for '{exp_name}', skipping.")
    if not exps:
        raise RuntimeError(f"No valid experiments found in {root_dir}")
    return exps


def plot_comparison(exps, output_path, tag='rewards', smooth_window=1):
    """
    Plot mean reward curves with shaded std across seeds, optionally smoothed.
    """
    sns.set()
    plt.figure(figsize=(10, 6))
    for exp_name, dfs in exps.items():
        # align steps to common max
        common_max = min(df.index.max() for df in dfs)
        all_steps = np.unique(np.concatenate([df.index.values for df in dfs]))
        common_steps = np.sort(all_steps[all_steps <= common_max])
        # build matrix of rewards
        mat = np.vstack([
            df['reward'].reindex(common_steps).interpolate(method='index').values
            for df in dfs
        ])
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        # apply running average if window > 1
        if smooth_window > 1:
            mean = pd.Series(mean).rolling(window=smooth_window, min_periods=1, center=True).mean().values
            std = pd.Series(std).rolling(window=smooth_window, min_periods=1, center=True).mean().values
        # plot with seaborn
        sns.lineplot(x=common_steps, y=mean, label=exp_name)
        plt.fill_between(common_steps, mean - std, mean + std, alpha=0.3)
    # format x-axis in millions
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x/1e6)}M" if x>=1e6 else f"{int(x)}"))
    plt.xlabel('Training Step')
    plt.ylabel(tag.capitalize())
    plt.title(f"Training {tag.capitalize()} Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Saved comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot reward curves for multiple experiments.')
    parser.add_argument('root_dir', type=str, help='Main folder with experiment subdirs')
    parser.add_argument('-t', '--tag', type=str, default='rewards', help='TensorBoard scalar tag')
    parser.add_argument('-o', '--output', type=str, default='reward_comparison.png', help='Output image')
    parser.add_argument('-w', '--smooth-window', type=int, default=1,
                        help='Window size for running average (1 = no smoothing)')
    args = parser.parse_args()
    exps = gather_experiment(args.root_dir, args.tag)
    plot_comparison(exps, args.output, args.tag, args.smooth_window)

if __name__ == '__main__':
    main()
