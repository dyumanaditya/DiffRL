#!/usr/bin/env python3
"""
rename_folders.py

Scan each subdirectory of TARGET_FOLDER for a .hydra/config.yaml, parse out:
  env.config.num_samples
  env.config.sigma
  env.config.contact.ke
  env.config.contact.kd

and rename the folder from F to:

  hopper_shac_bundle_{num_samples}_{sigma}_ke_{ke_str}_kd_{kd_str}__{F}

Usage:
  python rename_folders.py /path/to/target_folder [--dry-run]
"""

import os
import argparse
import yaml

def scientific_str(n):
    """
    Format a number like 700000 as '7e5', 1e6 as '1e6', etc.
    """
    s = f"{n:.0e}"       # e.g. '7e+05'
    # collapse the '+0' so '7e+05' → '7e5'
    return s.replace('e+0', 'e').replace('e+','e')

def main():
    parser = argparse.ArgumentParser(
        description="Rename subfolders based on their .hydra/config.yaml"
    )
    parser.add_argument(
        "target_folder",
        help="Directory containing the experiment folders to rename"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be renamed, but do not actually rename"
    )
    args = parser.parse_args()

    for name in os.listdir(args.target_folder):
        src = os.path.join(args.target_folder, name)
        if not os.path.isdir(src):
            continue

        cfg_path = os.path.join(src, ".hydra", "config.yaml")
        if not os.path.isfile(cfg_path):
            print(f"[SKIP] no config.yaml in {name}")
            continue

        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f)

        try:
            cfg = data["env"]["config"]
            ns = cfg["num_samples"]
            sig = cfg["sigma"]
            ke = cfg["contact"]["ke"]
            kd = cfg["contact"]["kd"]
        except KeyError as e:
            print(f"[ERROR] missing key {e} in {name}, skipping")
            continue

        ke_str = scientific_str(ke)
        kd_str = scientific_str(kd)
        new_name = (
            f"hopper_shac_bundle_{ns}_{sig}"
            f"_ke_{ke_str}_kd_{kd_str}__{name}"
        )
        dst = os.path.join(args.target_folder, new_name)

        print(f"RENAME: {name} → {new_name}")
        if not args.dry_run:
            os.rename(src, dst)

if __name__ == "__main__":
    main()

