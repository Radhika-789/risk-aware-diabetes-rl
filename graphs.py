
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "runs" / "dqn_risk_s0_300"
VAL_DIR = ROOT / "runs" / "eval_300_val"
TEST_DIR = ROOT / "runs" / "eval_300_test"

OUT_DIR = ROOT / "results" / "graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLICIES = ["cgm_rule", "best_300", "final_300", "announced_bb"]

# Moving-average window for the training curves.
SMOOTH_WINDOW = 15

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# ============================================================
# HELPERS
# ============================================================

def normalise_name(name):
    """Make CSV column names easier to match."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def read_csv(path):
    if not path.exists():
        print(f"[SKIP] File not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[SKIP] Could not read {path}: {exc}")
        return None

    if df.empty:
        print(f"[SKIP] CSV is empty: {path}")
        return None

    print(f"\n[READ] {path.relative_to(ROOT)}")
    print("       Columns:", list(df.columns))
    return df


def find_column(df, aliases):
    """Return the first column matching one of the alias names."""
    if df is None:
        return None

    lookup = {normalise_name(c): c for c in df.columns}

    for alias in aliases:
        key = normalise_name(alias)
        if key in lookup:
            return lookup[key]

    return None


def numeric_column(df, aliases):
    col = find_column(df, aliases)
    if col is None:
        return None, None

    values = pd.to_numeric(df[col], errors="coerce")
    if values.notna().sum() == 0:
        return None, col

    return values, col


def save_figure(fig, filename):
    path = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path.relative_to(ROOT)}")


def clean_policy(value):
    """Map common checkpoint and baseline names to standard labels."""
    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")

    if "announced" in text or "meal_announcement" in text:
        return "announced_bb"
    if "cgm" in text and "rule" in text:
        return "cgm_rule"
    if "best" in text and ("300" in text or "dqn" in text):
        return "best_300"
    if "final" in text and ("300" in text or "dqn" in text):
        return "final_300"

    # Exact common labels
    if text in POLICIES:
        return text

    return text


def get_policy_column(df):
    return find_column(
        df,
        [
            "policy", "label", "agent", "model", "controller",
            "method", "checkpoint", "algorithm", "name"
        ],
    )


def to_percent(series):
    """
    Keep percentage values as-is. Convert proportions in [0, 1]
    to percentages only if all non-null values are in that range.
    """
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()

    if len(valid) and valid.between(0, 1).all():
        return s * 100

    return s


def metric_aliases(metric):
    aliases = {
        "tir": [
            "tir_pct", "tir_percent", "tir", "time_in_range",
            "time_in_range_pct", "time_in_range_percent",
            "tir_70_180", "tir_70_180_pct",
        ],
        "tbr70": [
            "tbr70_pct", "tbr_70_pct", "tbr70", "tbr_70",
            "time_below_70", "time_below_70_pct",
            "below_70_pct", "below70_pct", "percent_below_70",
        ],
        "tbr54": [
            "tbr54_pct", "tbr_54_pct", "tbr54", "tbr_54",
            "time_below_54", "time_below_54_pct",
            "below_54_pct", "below54_pct", "percent_below_54",
        ],
        "tar180": [
            "tar180_pct", "tar_180_pct", "tar180", "tar_180",
            "time_above_180", "time_above_180_pct",
            "above_180_pct", "above180_pct", "percent_above_180",
        ],
        "tar250": [
            "tar250_pct", "tar_250_pct", "tar250", "tar_250",
            "time_above_250", "time_above_250_pct",
            "above_250_pct", "above250_pct", "percent_above_250",
        ],
        "return": [
            "episode_return", "return", "total_return",
            "mean_return", "avg_return", "reward", "episode_reward",
        ],
        "loss": [
            "loss", "train_loss", "training_loss", "mean_loss",
            "avg_loss", "q_loss", "td_loss",
        ],
        "episode": [
            "episode", "episode_num", "episode_number",
            "ep", "training_episode", "global_episode",
        ],
        "early_termination": [
            "early_termination_rate", "early_termination_pct",
            "terminated_early_pct", "early_terminated_pct",
            "early_termination", "terminated_early",
        ],
    }
    return aliases[metric]


def get_metric(df, metric):
    return numeric_column(df, metric_aliases(metric))


def maybe_convert_metric(df, metric):
    values, col = get_metric(df, metric)
    if values is None:
        return None, col
    if metric in {"tir", "tbr70", "tbr54", "tar180", "tar250",
                  "early_termination"}:
        values = to_percent(values)
    return values, col


# ============================================================
# GRAPHS 1-4: TRAINING AND VALIDATION
# ============================================================

def plot_training_return():
    df = read_csv(RUN_DIR / "train_log.csv")
    if df is None:
        return

    x, xcol = get_metric(df, "episode")
    y, ycol = get_metric(df, "return")

    if y is None:
        print("[SKIP] Graph 1: no recognised return column.")
        return

    if x is None:
        x = pd.Series(np.arange(1, len(df) + 1), index=df.index)
        xcol = "episode index"

    valid = pd.DataFrame({"episode": x, "return": y}).dropna()
    if valid.empty:
        return

    valid = valid.sort_values("episode")
    valid["moving_avg"] = valid["return"].rolling(
        SMOOTH_WINDOW, min_periods=1
    ).mean()

    fig, ax = plt.subplots()
    ax.plot(valid["episode"], valid["return"], alpha=0.3,
            linewidth=1, label="Episode return")
    ax.plot(valid["episode"], valid["moving_avg"], linewidth=2.2,
            label=f"{SMOOTH_WINDOW}-episode moving average")
    ax.set_title("1. DQN Training Return")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode return")
    ax.legend()
    save_figure(fig, "01_training_return.png")


def plot_training_tir():
    df = read_csv(RUN_DIR / "train_log.csv")
    if df is None:
        return

    x, _ = get_metric(df, "episode")
    y, ycol = maybe_convert_metric(df, "tir")

    if y is None:
        print("[SKIP] Graph 2: no recognised training TIR column.")
        return

    if x is None:
        x = pd.Series(np.arange(1, len(df) + 1), index=df.index)

    valid = pd.DataFrame({"episode": x, "tir": y}).dropna()
    valid = valid.sort_values("episode")
    valid["moving_avg"] = valid["tir"].rolling(
        SMOOTH_WINDOW, min_periods=1
    ).mean()

    fig, ax = plt.subplots()
    ax.plot(valid["episode"], valid["tir"], alpha=0.3,
            linewidth=1, label="Episode TIR")
    ax.plot(valid["episode"], valid["moving_avg"], linewidth=2.2,
            label=f"{SMOOTH_WINDOW}-episode moving average")
    ax.set_title("2. DQN Training Time in Range")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("TIR (%)")
    ax.set_ylim(bottom=0)
    ax.legend()
    save_figure(fig, "02_training_tir.png")


def plot_validation_curve():
    df = read_csv(RUN_DIR / "val_log.csv")
    if df is None:
        return

    x, _ = get_metric(df, "episode")
    tir, _ = maybe_convert_metric(df, "tir")
    ret, _ = get_metric(df, "return")

    if x is None:
        print("[SKIP] Graph 3: no recognised validation episode column.")
        return

    if tir is None and ret is None:
        print("[SKIP] Graph 3: validation TIR/return columns not found.")
        return

    fig, ax1 = plt.subplots()
    if tir is not None:
        data = pd.DataFrame({"x": x, "tir": tir}).dropna()
        ax1.plot(data["x"], data["tir"], marker="o",
                 linewidth=2, label="Validation TIR")
        ax1.set_ylabel("TIR (%)")
        ax1.set_ylim(bottom=0)

    if ret is not None:
        data = pd.DataFrame({"x": x, "ret": ret}).dropna()
        ax2 = ax1.twinx()
        ax2.plot(data["x"], data["ret"], marker="s",
                 linestyle="--", linewidth=1.8,
                 label="Validation return")
        ax2.set_ylabel("Validation return")

    ax1.set_title("3. Validation Performance During Training")
    ax1.set_xlabel("Training episode")

    lines = ax1.get_lines()
    labels = [line.get_label() for line in lines]
    if "ax2" in locals():
        lines2 = ax2.get_lines()
        lines += lines2
        labels += [line.get_label() for line in lines2]
    if lines:
        ax1.legend(lines, labels, loc="best")

    save_figure(fig, "03_validation_performance.png")


def plot_training_loss():
    df = read_csv(RUN_DIR / "train_log.csv")
    if df is None:
        return

    x, _ = get_metric(df, "episode")
    loss, losscol = get_metric(df, "loss")

    if loss is None:
        print(
            "[SKIP] Graph 4: no recognised loss column in train_log.csv. "
            "The training log may not save loss; we won't invent it."
        )
        return

    if x is None:
        x = pd.Series(np.arange(1, len(df) + 1), index=df.index)

    valid = pd.DataFrame({"episode": x, "loss": loss}).dropna()
    valid = valid.sort_values("episode")

    fig, ax = plt.subplots()
    ax.plot(valid["episode"], valid["loss"], alpha=0.45,
            linewidth=1, label="Logged loss")

    smooth = valid["loss"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
    ax.plot(valid["episode"], smooth, linewidth=2,
            label=f"{SMOOTH_WINDOW}-point moving average")

    ax.set_title("4. DQN Training Loss")
    ax.set_xlabel("Training episode / logged step")
    ax.set_ylabel("Loss")
    ax.legend()
    save_figure(fig, "04_training_loss.png")


# ============================================================
# EVALUATION SUMMARY NORMALISATION
# ============================================================

def load_evaluation_summary(folder):
    """
    Read summary.csv. If the expected policy/metric columns aren't
    recognised, print the columns so aliases can be adjusted.
    """
    df = read_csv(folder / "summary.csv")
    if df is None:
        return None

    policy_col = get_policy_column(df)
    if policy_col is None:
        print("[SKIP] No recognisable policy/label column in summary.csv.")
        return None

    df = df.copy()
    df["policy_std"] = df[policy_col].map(clean_policy)

    mapping = {}
    for metric in ["tir", "tbr70", "tbr54", "tar180", "tar250",
                   "return", "early_termination"]:
        values, original_col = maybe_convert_metric(df, metric)
        if values is not None:
            df[metric] = values
            mapping[metric] = original_col

    if not mapping:
        print("[SKIP] No recognised metrics in summary.csv.")
        return None

    print("       Recognised evaluation metrics:", mapping)
    return df


def plot_policy_metric(df, metric, title, ylabel, filename):
    if df is None or metric not in df.columns:
        print(f"[SKIP] {filename}: metric '{metric}' not available.")
        return

    data = df[df["policy_std"].isin(POLICIES)].copy()
    data = data.dropna(subset=[metric])

    if data.empty:
        print(f"[SKIP] {filename}: none of the requested policies found.")
        print("       Policy values found:", sorted(df["policy_std"].unique()))
        return

    # If there are duplicate rows per policy, retain the provided summary
    # rows as-is rather than silently averaging different groups.
    labels = data["policy_std"].tolist()
    values = data[metric].tolist()

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(bottom=0)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )

    save_figure(fig, filename)


def plot_glucose_range_stacked(df, filename):
    if df is None:
        return

    needed = ["tir", "tbr70", "tar180"]
    if not all(m in df.columns for m in needed):
        print(
            f"[SKIP] {filename}: requires TIR, TBR<70 and TAR>180 "
            "columns in summary.csv."
        )
        return

    data = df[df["policy_std"].isin(POLICIES)].copy()
    data = data.dropna(subset=needed)

    if data.empty:
        print(f"[SKIP] {filename}: no requested policies found.")
        return

    # These three ranges are mutually exclusive when calculated from
    # the same glucose samples and conventionally should sum to ~100%.
    fig, ax = plt.subplots()
    bottom = np.zeros(len(data))

    components = [
        ("tbr70", "TBR: <70 mg/dL"),
        ("tir", "TIR: 70–180 mg/dL"),
        ("tar180", "TAR: >180 mg/dL"),
    ]

    for metric, label in components:
        vals = data[metric].to_numpy(dtype=float)
        ax.bar(data["policy_std"], vals, bottom=bottom, label=label)
        bottom += vals

    ax.set_title("10. Glucose-Range Distribution by Policy")
    ax.set_ylabel("Time (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.tick_params(axis="x", rotation=15)

    # Warn if percentages do not add to approximately 100.
    totals = data[needed].sum(axis=1)
    for policy, total in zip(data["policy_std"], totals):
        if abs(total - 100) > 3:
            warnings.warn(
                f"{policy}: TBR70 + TIR + TAR180 = {total:.1f}%, "
                "not approximately 100%. Verify metric definitions."
            )

    save_figure(fig, filename)


def plot_evaluation_graphs(folder, split_label):
    df = load_evaluation_summary(folder)
    if df is None:
        return

    prefix = f"{split_label}_"

    # Graph 5: TIR
    plot_policy_metric(
        df, "tir",
        f"5. TIR Comparison — {split_label.title()}",
        "Time in range (%)",
        prefix + "05_tir_comparison.png",
    )

    # Graph 6: TAR >180
    plot_policy_metric(
        df, "tar180",
        f"6. TAR Comparison (>180 mg/dL) — {split_label.title()}",
        "Time above 180 mg/dL (%)",
        prefix + "06_tar_comparison.png",
    )

    # Graph 7: TBR <70
    plot_policy_metric(
        df, "tbr70",
        f"7. TBR Comparison (<70 mg/dL) — {split_label.title()}",
        "Time below 70 mg/dL (%)",
        prefix + "07_tbr_comparison.png",
    )

    # Graph 8: severe hypoglycemia <54
    plot_policy_metric(
        df, "tbr54",
        f"8. Severe Hypoglycemia (<54 mg/dL) — {split_label.title()}",
        "Time below 54 mg/dL (%)",
        prefix + "08_severe_hypoglycemia.png",
    )

    # Graph 9: severe hyperglycemia >250
    plot_policy_metric(
        df, "tar250",
        f"9. Severe Hyperglycemia (>250 mg/dL) — {split_label.title()}",
        "Time above 250 mg/dL (%)",
        prefix + "09_severe_hyperglycemia.png",
    )

    # Graph 10: combined glucose ranges
    plot_glucose_range_stacked(df, prefix + "10_glucose_range_distribution.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("GlucoGuard AI — research plotting script")
    print("Project root:", ROOT)
    print("Graph output:", OUT_DIR)
    print("=" * 65)

    # Training graphs 1–4
    plot_training_return()
    plot_training_tir()
    plot_validation_curve()
    plot_training_loss()

    # Evaluation graphs 5–10 for validation and test separately
    plot_evaluation_graphs(VAL_DIR, "validation")
    plot_evaluation_graphs(TEST_DIR, "test")

    print("\nFinished.")
    print("Check the PNGs in:", OUT_DIR)
    print(
        "\nIf a graph was skipped, read the [SKIP] message above. "
        "Column names vary between project versions; use the printed "
        "CSV columns to update the aliases instead of guessing values."
    )


if __name__ == "__main__":
    main()
