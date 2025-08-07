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


def load_rewards(run_path, tag='rewards'):
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


def plot_three_runs(paths, tag, smooth_window, output_path):
    """
    Plot reward curves for three single-run folders with optional smoothing.
    """
    sns.set()
    plt.figure(figsize=(10, 6))
    labels = [os.path.basename(p.rstrip('/')) for p in paths]
    for path, label in zip(paths, labels):
        df = load_rewards(path, tag)
        steps = df.index.values
        values = df['reward'].values
        # apply rolling average if requested
        if smooth_window > 1:
            values = pd.Series(values).rolling(window=smooth_window, min_periods=1, center=True).mean().values
        sns.lineplot(x=steps, y=values, label=label)

    # format x-axis ticks in millions
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x)}"))
    plt.xlabel('Training Step')
    plt.ylabel(tag.capitalize())
    plt.title(f"Training {tag.capitalize()} Curves for Three Runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Saved plot with three runs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot three single-run reward curves with smoothing.')
    parser.add_argument('run_dirs', nargs=3, type=str,
                        help='Paths to three run folders')
    parser.add_argument('-t', '--tag', type=str, default='rewards',
                        help='TensorBoard scalar tag to plot')
    parser.add_argument('-w', '--smooth-window', type=int, default=1,
                        help='Window size for running average (1 = no smoothing)')
    parser.add_argument('-o', '--output', type=str, default='three_runs_comparison.png',
                        help='Path for output image')
    args = parser.parse_args()
    # verify directories
    for p in args.run_dirs:
        if not os.path.isdir(p):
            parser.error(f"Run directory not found: {p}")
    plot_three_runs(args.run_dirs, args.tag, args.smooth_window, args.output)

if __name__ == '__main__':
    main()
