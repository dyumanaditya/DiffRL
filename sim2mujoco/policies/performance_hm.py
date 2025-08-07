import os
import glob
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def extract_performance(run_path, last_k=1000):
    """
    Extracts the average of the last `last_k` reward values from TensorBoard logs in a run folder,
    along with the num_samples and sigma values from the config.
    """
    # Load config file
    config_file = os.path.join(run_path, '.hydra', 'config.yaml')
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    # Read parameters
    try:
        num_samples = cfg['env']['config']['num_samples']
        sigma = cfg['env']['config']['sigma']
    except KeyError:
        raise ValueError(f"Missing num_samples or sigma in config at {config_file}")

    # Locate TensorBoard event files
    log_dir = os.path.join(run_path, 'logs', 'log')
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    # Initialize EventAccumulator
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            # keep all scalar data for 'rewards'
            event_accumulator.SCALARS: 0,
        }
    )
    ea.Reload()

    # Check for 'rewards' tag
    tags = ea.Tags().get('scalars', [])
    if 'rewards' not in tags:
        raise KeyError(f"'rewards' scalar not found in TensorBoard logs at {log_dir}")
    events = ea.Scalars('rewards')
    values = [e.value for e in events]

    # Compute average of last_k values
    if len(values) == 0:
        raise ValueError(f"No reward values found in logs at {log_dir}")
    avg_perf = np.mean(values[-last_k:]) if len(values) >= last_k else np.mean(values)

    return num_samples, sigma, avg_perf


def gather_results(root_dir):
    """
    Walks through each run folder in root_dir, extracts performance metrics,
    and returns a DataFrame.
    """
    records = []
    for entry in os.listdir(root_dir):
        run_path = os.path.join(root_dir, entry)
        if not os.path.isdir(run_path):
            continue
        try:
            n, s, p = extract_performance(run_path)
            records.append({'num_samples': n, 'sigma': s, 'performance': p})
        except Exception as e:
            print(f"Warning: could not process {run_path}: {e}")

    if not records:
        raise RuntimeError(f"No valid runs found in {root_dir}")
    return pd.DataFrame.from_records(records)


def plot_heatmap(df, output_path, log_scale=True):
    """
    Creates and saves a heatmap of performance over num_samples (x-axis) and sigma (y-axis).
    """
    # Pivot for heatmap
    pivot = df.pivot(index='sigma', columns='num_samples', values='performance')
    x = pivot.columns.values
    y = pivot.index.values
    z = pivot.values

    plt.figure(figsize=(8, 6))
    # Show as image; extent sets axis limits
    im = plt.imshow(
        z,
        origin='lower',
        aspect='auto',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='viridis'
    )
    if log_scale:
        plt.xscale('log')
        plt.xlabel('num_samples (log scale)')
    else:
        plt.xlabel('num_samples')
    plt.ylabel('sigma')
    plt.title('Performance Heatmap')
    cbar = plt.colorbar(im)
    cbar.set_label('Average Reward')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Heatmap saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate a performance heatmap from TensorBoard runs.'
    )
    parser.add_argument(
        'root_dir',
        type=str,
        help='Path to the directory containing run subfolders.'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='performance_heatmap.png',
        help='Output path for the saved heatmap image.'
    )
    parser.add_argument(
        '--no-log',
        action='store_true',
        help='Disable log-scaling on the num_samples axis.'
    )
    args = parser.parse_args()

    df = gather_results(args.root_dir)
    plot_heatmap(df, args.output, log_scale=not args.no_log)


if __name__ == '__main__':
    main()
