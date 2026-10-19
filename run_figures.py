"""Command line entry point.

Examples
    python run_figures.py --data /path/to/zenodo --out results fig2
    python run_figures.py --data /path/to/zenodo --out results all
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, no window needed

import figures  # noqa: E402

TARGETS = {
    "fig1": [figures.fig1_trajectories, figures.fig1_snapshots],
    "fig2": [figures.fig2_phase_slips],
    "fig3": [figures.fig3_dispersive_shock],
    "fig5": [figures.fig5_background_flow],
    "supp2": [figures.supp_n2_isat],
    "supp3": [figures.supp_healing_length],
    "supp5": [figures.supp_flow_patterns],
}


def main(argv=None):
    p = argparse.ArgumentParser(description="Reproduce the figures of the article.")
    p.add_argument("target", choices=[*TARGETS, "all"], help="figure to produce")
    p.add_argument("--data", required=True, help="folder containing the dataset folders")
    p.add_argument("--out", default="results", help="output folder (default: results)")
    args = p.parse_args(argv)

    names = list(TARGETS) if args.target == "all" else [args.target]
    failed = []
    for name in names:
        for func in TARGETS[name]:
            print(f"[{name}] {func.__name__} ...", flush=True)
            try:
                func(args.data, args.out)
            except FileNotFoundError as e:  # missing dataset: keep going with the others
                print(f"   skipped: {e}")
                failed.append(name)
    print(f"Figures written to {args.out!r}.")
    return 1 if failed and args.target != "all" else 0


if __name__ == "__main__":
    sys.exit(main())
