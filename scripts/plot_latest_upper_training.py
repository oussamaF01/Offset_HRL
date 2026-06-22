#!/usr/bin/env python3
"""Plot the newest upper-PPO training run.

The training CSV can be several gigabytes because it contains nested JSON
diagnostics. This script reads only the numeric columns required for plots.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GNB_IDS = (0, 1, 2)
SLICE_TYPES = ("eMBB", "URLLC", "mMTC")
COLORS = ("tab:blue", "tab:orange", "tab:green")

CORE_COLUMNS = [
    "step",
    "episode",
    "episode_step",
    "reward",
    "episode_return",
    "done",
    "scenario_name",
    "load_variance",
    "target_load_error",
    "overload_ratio",
    "sla_count",
    "sla_severity",
    "handover_count",
    "load_imbalance_start",
    "load_imbalance_end",
    "dense_window_reward",
    "global_network_cost",
    "global_cost_improvement",
    "global_action_penalty",
    "global_negative_bias_penalty",
    "global_bad_direction_penalty",
    "reward_load_improvement",
    "reward_saturation_improvement",
    "reward_sla_improvement",
    "reward_neutral_bias_penalty",
    "reward_wrong_bias_penalty",
    "reward_sla_severity_level_penalty",
    "reward_load_balance_level_bonus",
    "saturation_count",
    "overloaded_negative_fraction",
    "light_nonnegative_fraction",
    "network_throughput_mbps",
    "network_offered_mbps",
    "network_delivery_ratio",
    "network_completed_delay_ms",
    "network_mean_hol_delay_ms",
    "network_max_hol_delay_ms",
    "network_queue_kbits",
    "network_drop_ratio",
    "network_packet_failure_ratio",
]

MATRIX_COLUMNS = [
    f"{prefix}_g{gnb_id}_{slice_type}"
    for prefix in ("bias", "target_load", "load", "sla", "ue_count")
    for gnb_id in GNB_IDS
    for slice_type in SLICE_TYPES
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create analysis plots for the latest upper-PPO training run."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("models/upper_ppo_3gnb"),
        help="Directory containing run_*/training_log.csv.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific run directory. By default, use the newest training log.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <run-dir>/plots.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=500,
        help="Number of logged PPO steps used for rolling means.",
    )
    parser.add_argument(
        "--episode-window",
        type=int,
        default=100,
        help="Number of episodes used for episode-level rolling means.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20_000,
        help="Maximum points drawn per line.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def find_run(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        run_dir = args.run_dir
        csv_path = run_dir / "training_log.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing training log: {csv_path}")
        return run_dir

    candidates = list(args.runs_root.glob("run_*/training_log.csv"))
    if not candidates:
        raise FileNotFoundError(f"No run_*/training_log.csv found under {args.runs_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def read_training_log(csv_path: Path) -> pd.DataFrame:
    available = pd.read_csv(csv_path, nrows=0).columns.tolist()
    wanted = [column for column in CORE_COLUMNS + MATRIX_COLUMNS if column in available]
    chunks = pd.read_csv(
        csv_path,
        usecols=wanted,
        chunksize=50_000,
        low_memory=False,
        on_bad_lines="skip",
    )
    frame = pd.concat(chunks, ignore_index=True)
    if frame.empty:
        raise ValueError(f"No readable training rows in {csv_path}")

    for column in frame.columns:
        if column not in {"scenario_name", "done"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["done"] = frame.get("done", False).astype(str).str.lower().eq("true")
    frame = frame.dropna(subset=["step"]).sort_values("step").reset_index(drop=True)
    return frame


def sampled(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    stride = max(1, int(np.ceil(len(frame) / max(max_points, 1))))
    return frame.iloc[::stride]


def rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(max(1, int(window)), min_periods=1).mean()


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(path)


def plot_overview(
    frame: pd.DataFrame,
    output_dir: Path,
    rolling_window: int,
    max_points: int,
    dpi: int,
) -> None:
    view = sampled(frame, max_points)
    x = view["step"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    fig.suptitle("Upper PPO training overview", fontsize=16)

    for column, label, color in (
        ("dense_window_reward", "Dense reward", "tab:blue"),
        ("reward", "PPO reward", "tab:gray"),
    ):
        if column in frame:
            axes[0, 0].plot(
                x,
                rolling(frame[column], rolling_window).iloc[view.index],
                label=label,
                color=color,
                linewidth=1.2,
            )
    axes[0, 0].axhline(0.0, color="black", linewidth=0.7)
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()

    for column, label in (
        ("load_variance", "Load variance"),
        ("global_network_cost", "Network cost"),
        ("target_load_error", "Target/load error"),
    ):
        if column in frame:
            axes[0, 1].plot(
                x,
                rolling(frame[column], rolling_window).iloc[view.index],
                label=label,
                linewidth=1.1,
            )
    axes[0, 1].set_ylabel("Cost")
    axes[0, 1].legend()

    for column, label in (
        ("sla_severity", "SLA severity"),
        ("sla_count", "SLA count"),
        ("saturation_count", "Saturated pairs"),
    ):
        if column in frame:
            axes[1, 0].plot(
                x,
                rolling(frame[column], rolling_window).iloc[view.index],
                label=label,
                linewidth=1.1,
            )
    axes[1, 0].set_ylabel("SLA / saturation")
    axes[1, 0].legend()

    if "handover_count" in frame:
        axes[1, 1].plot(
            x,
            rolling(frame["handover_count"], rolling_window).iloc[view.index],
            color="tab:purple",
            label="Handovers per PPO step",
        )
    if "overload_ratio" in frame:
        axes[1, 1].plot(
            x,
            rolling(frame["overload_ratio"], rolling_window).iloc[view.index],
            color="tab:red",
            label="Overload ratio",
        )
    axes[1, 1].set_ylabel("Mobility / overload")
    axes[1, 1].legend()

    for column, label in (
        ("overloaded_negative_fraction", "Correct overloaded bias"),
        ("light_nonnegative_fraction", "Correct light-cell bias"),
        ("network_delivery_ratio", "Delivery ratio"),
    ):
        if column in frame:
            axes[2, 0].plot(
                x,
                rolling(frame[column], rolling_window).iloc[view.index],
                label=label,
            )
    axes[2, 0].set_ylim(-0.05, 1.05)
    axes[2, 0].set_ylabel("Fraction")
    axes[2, 0].legend()

    for column, label in (
        ("network_throughput_mbps", "Throughput"),
        ("network_offered_mbps", "Offered"),
    ):
        if column in frame:
            axes[2, 1].plot(
                x,
                rolling(frame[column], rolling_window).iloc[view.index],
                label=label,
            )
    axes[2, 1].set_ylabel("Mbit/s")
    axes[2, 1].legend()

    for axis in axes[-1]:
        axis.set_xlabel("PPO step")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    save_figure(fig, output_dir / "01_training_overview.png", dpi)


def active_slices(frame: pd.DataFrame) -> list[str]:
    active = []
    for slice_type in SLICE_TYPES:
        columns = [
            f"{prefix}_g{gnb_id}_{slice_type}"
            for prefix in ("target_load", "load", "ue_count")
            for gnb_id in GNB_IDS
            if f"{prefix}_g{gnb_id}_{slice_type}" in frame
        ]
        if columns and frame[columns].fillna(0.0).abs().to_numpy().max() > 1e-9:
            active.append(slice_type)
    return active or list(SLICE_TYPES)


def plot_slice_dynamics(
    frame: pd.DataFrame,
    output_dir: Path,
    rolling_window: int,
    max_points: int,
    dpi: int,
) -> None:
    view = sampled(frame, max_points)
    x = view["step"]
    for slice_type in active_slices(frame):
        fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
        fig.suptitle(f"{slice_type}: gNB load, upper bias, and SLA", fontsize=15)

        for gnb_id, color in zip(GNB_IDS, COLORS):
            load_column = f"load_g{gnb_id}_{slice_type}"
            target_column = f"target_load_g{gnb_id}_{slice_type}"
            bias_column = f"bias_g{gnb_id}_{slice_type}"
            sla_column = f"sla_g{gnb_id}_{slice_type}"
            if load_column in frame:
                axes[0].plot(
                    x,
                    rolling(frame[load_column], rolling_window).iloc[view.index],
                    color=color,
                    label=f"gNB{gnb_id} load",
                )
            if target_column in frame:
                axes[0].plot(
                    x,
                    rolling(frame[target_column], rolling_window).iloc[view.index],
                    color=color,
                    linestyle=":",
                    alpha=0.65,
                    label=f"gNB{gnb_id} scenario target",
                )
            if bias_column in frame:
                axes[1].plot(
                    x,
                    rolling(frame[bias_column], rolling_window).iloc[view.index],
                    color=color,
                    label=f"gNB{gnb_id}",
                )
            if sla_column in frame:
                axes[2].plot(
                    x,
                    rolling(frame[sla_column], rolling_window).iloc[view.index],
                    color=color,
                    label=f"gNB{gnb_id}",
                )

        axes[0].set_ylabel("PRB load")
        axes[0].set_ylim(bottom=-0.05)
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_ylabel("Upper bias")
        axes[1].set_ylim(-1.05, 1.05)
        axes[2].set_ylabel("SLA severity")
        axes[2].set_xlabel("PPO step")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(ncol=3)
        save_figure(
            fig,
            output_dir / f"02_{slice_type.lower()}_load_bias_sla.png",
            dpi,
        )


def plot_reward_components(
    frame: pd.DataFrame,
    output_dir: Path,
    rolling_window: int,
    max_points: int,
    dpi: int,
) -> None:
    view = sampled(frame, max_points)
    x = view["step"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    fig.suptitle("Upper PPO reward decomposition", fontsize=15)

    positive = (
        ("reward_load_improvement", "Load improvement"),
        ("reward_load_balance_level_bonus", "Persistent load-balance bonus"),
        ("reward_saturation_improvement", "Saturation improvement"),
        ("reward_sla_improvement", "SLA improvement"),
        ("global_cost_improvement", "Total cost improvement"),
    )
    penalties = (
        ("global_action_penalty", "Action smoothness"),
        ("global_bad_direction_penalty", "Bad direction"),
        ("reward_neutral_bias_penalty", "Neutral-bias raw penalty"),
        ("reward_wrong_bias_penalty", "Wrong-bias raw penalty"),
        ("global_negative_bias_penalty", "Negative-bias magnitude penalty"),
        ("reward_sla_severity_level_penalty", "Persistent SLA"),
    )
    for column, label in positive:
        if column in frame:
            axes[0].plot(
                x,
                rolling(frame[column], rolling_window).iloc[view.index],
                label=label,
            )
    for column, label in penalties:
        if column in frame:
            axes[1].plot(
                x,
                rolling(frame[column], rolling_window).iloc[view.index],
                label=label,
            )
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Reward contribution")
    axes[1].set_ylabel("Penalty magnitude")
    axes[1].set_xlabel("PPO step")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=2)
    save_figure(fig, output_dir / "03_reward_components.png", dpi)


def episode_summary(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "last_step": ("step", "max"),
        "episode_return": ("episode_return", "last"),
        "mean_dense_reward": ("dense_window_reward", "mean"),
        "mean_load_variance": ("load_variance", "mean"),
        "final_load_variance": ("load_variance", "last"),
        "mean_sla_severity": ("sla_severity", "mean"),
        "handovers": ("handover_count", "sum"),
        "scenario_name": ("scenario_name", "last"),
    }
    valid = {
        output: (column, operation)
        for output, (column, operation) in aggregations.items()
        if column in frame
    }
    return frame.groupby("episode", as_index=False).agg(**valid)


def plot_episode_trends(
    episodes: pd.DataFrame,
    output_dir: Path,
    episode_window: int,
    max_points: int,
    dpi: int,
) -> None:
    view = sampled(episodes, max_points)
    x = view["episode"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    fig.suptitle("Episode-level learning trends", fontsize=15)
    panels = (
        ("episode_return", "Episode return"),
        ("mean_load_variance", "Mean load variance"),
        ("mean_sla_severity", "Mean SLA severity"),
        ("handovers", "Handovers per episode"),
    )
    for axis, (column, label) in zip(axes.flat, panels):
        if column in episodes:
            axis.plot(
                x,
                rolling(episodes[column], episode_window).iloc[view.index],
                color="tab:blue",
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Episode")
    save_figure(fig, output_dir / "04_episode_trends.png", dpi)


def write_summary(
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    run_dir: Path,
    output_dir: Path,
) -> None:
    tail_rows = max(1, min(len(frame), 5_000))
    tail = frame.tail(tail_rows)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    summary = {
        "run_dir": str(run_dir),
        "rows_read": int(len(frame)),
        "last_step": int(frame["step"].iloc[-1]),
        "episodes_seen": int(frame["episode"].nunique()),
        "completed_episode_rows": int(frame["done"].sum()),
        "scenarios": {
            str(key): int(value)
            for key, value in frame["scenario_name"].value_counts().items()
        },
        "tail_rows": tail_rows,
        "tail_mean_dense_reward": float(tail["dense_window_reward"].mean()),
        "tail_mean_load_variance": float(tail["load_variance"].mean()),
        "tail_mean_sla_severity": float(tail["sla_severity"].mean()),
        "tail_mean_handovers_per_step": float(tail["handover_count"].mean()),
        "tail_mean_overloaded_negative_fraction": float(
            tail["overloaded_negative_fraction"].mean()
        ),
        "active_slices": active_slices(frame),
        "configured_total_timesteps": config.get("total_timesteps"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    episodes.to_csv(output_dir / "episode_summary.csv", index=False)
    print(output_dir / "summary.json")
    print(output_dir / "episode_summary.csv")


def main() -> None:
    args = parse_args()
    run_dir = find_run(args)
    output_dir = args.output_dir or run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "training_log.csv"

    print(f"Reading selected columns from {csv_path}")
    frame = read_training_log(csv_path)
    episodes = episode_summary(frame)
    print(
        f"Loaded {len(frame):,} rows through PPO step "
        f"{int(frame['step'].iloc[-1]):,}; episodes={len(episodes):,}"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_overview(
        frame, output_dir, args.rolling_window, args.max_points, args.dpi
    )
    plot_slice_dynamics(
        frame, output_dir, args.rolling_window, args.max_points, args.dpi
    )
    plot_reward_components(
        frame, output_dir, args.rolling_window, args.max_points, args.dpi
    )
    plot_episode_trends(
        episodes, output_dir, args.episode_window, args.max_points, args.dpi
    )
    write_summary(frame, episodes, run_dir, output_dir)


if __name__ == "__main__":
    main()
