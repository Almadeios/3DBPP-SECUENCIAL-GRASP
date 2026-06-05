import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_RESULTS_DIR = os.path.join(ROOT_DIR, "resultados")
BASE_OUTPUT_DIR = os.path.join(ROOT_DIR, "results_analysis")
FIGURES_DIR = os.path.join(BASE_OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(BASE_OUTPUT_DIR, "tables")
DATASETS = ["blockout", "kitchen"]
FIGURE_FILENAMES = {
    "fill_vs_step": "fill_vs_step.png",
    "runtime_vs_k": "runtime_vs_k.png",
    "convergence": "convergence.png",
    "objects_vs_step": "objects_vs_step.png",
    "fill_heatmap": "fill_heatmap.png",
    "runtime_heatmap": "runtime_heatmap.png",
    "placed_heatmap": "placed_heatmap.png",
}

plt.style.use("ggplot")

# ============================================================
# HELPERS
# ============================================================

def ensure_output_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


def smooth_curve(x, y, points=200):
    x = np.array(x)
    y = np.array(y)

    if len(x) < 3:
        return x, y

    x_smooth = np.linspace(x.min(), x.max(), points)
    spline = make_interp_spline(x, y, k=2)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth


def parse_filename_metadata(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r"_k(\d+)_s(\d+)", filename)
    if not match:
        return None, None

    k = int(match.group(1))
    step_digits = match.group(2)
    step_value = float(step_digits) / 1000.0
    return k, step_value


def save_figure(fig, name):
    path = os.path.join(FIGURES_DIR, FIGURE_FILENAMES[name])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")


def plot_metric_heatmap(df, value_col, title, filename, cmap="viridis_r"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = df[df["dataset"] == dataset]
        if subset.empty:
            continue

        pivot = (
            subset.groupby(["step", "k"])[value_col]
            .mean()
            .unstack("k")
        )
        pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            cmap=cmap,
        )

        ax.set_title(f"{dataset.capitalize()}: {title}")
        ax.set_xlabel("Buffer Size (k)")
        ax.set_ylabel("Step Size")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(np.round(pivot.index, 3))
        fig.colorbar(im, ax=ax, label=value_col.replace("_", " ").title())

    plt.tight_layout()
    save_figure(fig, filename)
    plt.show()


def compute_pareto_frontier(summary):
    if summary.empty:
        return summary

    candidate = summary.sort_values(["mean_runtime", "mean_fill"], ascending=[True, False])
    frontier = []
    best_fill = -np.inf

    for _, row in candidate.iterrows():
        if row["mean_fill"] > best_fill:
            frontier.append(row)
            best_fill = row["mean_fill"]

    return pd.DataFrame(frontier)


def plot_pareto_frontier(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    for dataset in DATASETS:
        summary = create_summary(df, dataset)
        if summary.empty:
            continue

        ax.scatter(
            summary["mean_runtime"],
            summary["mean_fill"],
            s=100,
            alpha=0.5,
            label=f"{dataset} combos",
        )

        pareto = compute_pareto_frontier(summary)
        if not pareto.empty:
            ax.plot(
                pareto["mean_runtime"],
                pareto["mean_fill"],
                marker="o",
                linewidth=2,
                markersize=8,
                label=f"{dataset} Pareto",
            )
            for _, row in pareto.iterrows():
                ax.annotate(
                    f"k={row['k']} s={row['step']}",
                    (row["mean_runtime"], row["mean_fill"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

    ax.set_title("Pareto Frontier: Fill vs Runtime")
    ax.set_xlabel("Mean Runtime (s)")
    ax.set_ylabel("Mean Fill Percentage")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_placed_heatmap(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = df[df["dataset"] == dataset]
        if subset.empty:
            continue

        pivot = (
            subset.groupby(["step", "k"])["placed"]
            .mean()
            .unstack("k")
        )
        pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            cmap="viridis_r",
        )

        ax.set_title(f"{dataset.capitalize()}: Objects Packed Heatmap")
        ax.set_xlabel("Buffer Size (k)")
        ax.set_ylabel("Step Size")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(np.round(pivot.index, 3))
        fig.colorbar(im, ax=ax, label="Objects Packed")

    plt.tight_layout()
    save_figure(fig, "placed_heatmap")
    plt.show()


def load_meta_results():
    rows = []
    grasp_rows = []

    for dataset in DATASETS:
        meta_dir = os.path.join(BASE_RESULTS_DIR, dataset, "meta")
        print(f"\nSearching in: {meta_dir}")

        if not os.path.isdir(meta_dir):
            print(f"Directory not found: {meta_dir}")
            continue

        json_files = []
        for root, _, files in os.walk(meta_dir):
            for file_name in sorted(files):
                if file_name.endswith(".json") and file_name.startswith("meta_"):
                    json_files.append(os.path.join(root, file_name))

        print(f"Found {len(json_files)} JSON files")

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                print(f"ERROR loading {file_path}: {exc}")
                continue

            k = data.get("buffer_size")
            step = data.get("step")
            parsed_k, parsed_step = parse_filename_metadata(file_path)
            if k is None:
                k = parsed_k
            if step is None and parsed_step is not None:
                step = parsed_step

            if k is None or step is None:
                print(f"Skipping {file_path}: missing buffer_size or step")
                continue

            row = {
                "dataset": data.get("dataset", dataset),
                "k": int(k),
                "step": float(step),
                "placed": data.get("placed", np.nan),
                "fill_percent": data.get("fill_percent", np.nan),
                "elapsed_total": data.get("elapsed_total", np.nan),
            }

            for optional_key in ["sequence_index", "elapsed_packing", "grasp_iterations"]:
                if optional_key in data:
                    row[optional_key] = data[optional_key]

            rows.append(row)

            iterations_data = data.get("grasp_iterations_data")
            if isinstance(iterations_data, list):
                for i, iteration in enumerate(iterations_data):
                    if not isinstance(iteration, (list, tuple)) or len(iteration) < 3:
                        continue
                    grasp_rows.append({
                        "dataset": data.get("dataset", dataset),
                        "k": int(k),
                        "step": float(step),
                        "grasp_iter": i + 1,
                        "iter_elapsed": iteration[0],
                        "iter_placed": iteration[1],
                        "iter_fill": iteration[2],
                    })

    df = pd.DataFrame(rows)
    grasp_df = pd.DataFrame(grasp_rows)
    return df, grasp_df


def save_raw_tables(df):
    raw_tables = {}
    for dataset in DATASETS:
        subset = df[df["dataset"] == dataset]
        raw_df = subset[["dataset", "k", "step", "placed", "fill_percent", "elapsed_total"]].copy()
        raw_path = os.path.join(TABLES_DIR, f"{dataset}_results.csv")
        raw_df.to_csv(raw_path, index=False)
        print(f"Saved table: {raw_path}")
        raw_tables[dataset] = raw_df
    return raw_tables


def create_summary(df, dataset):
    subset = df[df["dataset"] == dataset]
    if subset.empty:
        return pd.DataFrame(
            columns=["k", "step", "mean_fill", "std_fill", "mean_placed", "std_placed", "mean_runtime", "std_runtime"]
        )
    summary = (
        subset.groupby(["k", "step"]).agg(
            mean_fill=("fill_percent", "mean"),
            std_fill=("fill_percent", "std"),
            mean_placed=("placed", "mean"),
            std_placed=("placed", "std"),
            mean_runtime=("elapsed_total", "mean"),
            std_runtime=("elapsed_total", "std"),
        ).reset_index()
    )
    summary["std_fill"] = summary["std_fill"].fillna(0)
    summary["std_placed"] = summary["std_placed"].fillna(0)
    summary["std_runtime"] = summary["std_runtime"].fillna(0)
    return summary


def save_summary_tables(df):
    summaries = {}
    for dataset in DATASETS:
        summary = create_summary(df, dataset)
        summary_path = os.path.join(TABLES_DIR, f"{dataset}_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")
        summaries[dataset] = summary
    return summaries


def save_top5_tables(df):
    top5_tables = {}
    for dataset in DATASETS:
        summary = create_summary(df, dataset)
        if summary.empty:
            top5 = summary
        else:
            top5 = summary.sort_values(
                by=["mean_fill", "mean_runtime"],
                ascending=[False, True]
            ).head(5)
            top5["dataset"] = dataset
            top5 = top5[["dataset", "k", "step", "mean_fill", "std_fill", "mean_placed", "std_placed", "mean_runtime", "std_runtime"]]

        top5_path = os.path.join(TABLES_DIR, f"{dataset}_top5.csv")
        top5.to_csv(top5_path, index=False)
        print(f"Saved top5: {top5_path}")
        top5_tables[dataset] = top5
    return top5_tables


def write_excel(raw_tables, summaries, top5_tables):
    excel_path = os.path.join(BASE_OUTPUT_DIR, "results_analysis.xlsx")
    engine = None
    try:
        import openpyxl  # noqa: F401
        engine = "openpyxl"
    except ImportError:
        try:
            import xlsxwriter  # noqa: F401
            engine = "xlsxwriter"
        except ImportError:
            engine = None
    if engine is None:
        print("WARNING: No Excel engine installed. Install openpyxl or xlsxwriter to save results_analysis.xlsx.")
        return

    with pd.ExcelWriter(excel_path, engine=engine) as writer:
        raw_tables["blockout"].to_excel(writer, sheet_name="Blockout_Raw", index=False)
        raw_tables["kitchen"].to_excel(writer, sheet_name="Kitchen_Raw", index=False)
        summaries["blockout"].to_excel(writer, sheet_name="Blockout_Summary", index=False)
        summaries["kitchen"].to_excel(writer, sheet_name="Kitchen_Summary", index=False)
        top5_tables["blockout"].to_excel(writer, sheet_name="Blockout_Top5", index=False)
        top5_tables["kitchen"].to_excel(writer, sheet_name="Kitchen_Top5", index=False)

    print(f"Saved Excel workbook: {excel_path}")


def plot_fill_vs_step(df, k_order, step_order):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = df[df["dataset"] == dataset]
        for k in k_order:
            temp = subset[subset["k"] == k]
            grouped = temp.groupby("step")["fill_percent"].mean().reindex(step_order)
            x = grouped.index.values
            y = grouped.values
            x_smooth, y_smooth = smooth_curve(x, y)
            ax.plot(x_smooth, y_smooth, linewidth=3, alpha=0.9, label=f"k={k}")
            ax.scatter(x, y, s=80, zorder=3)
        ax.set_title(f"{dataset.capitalize()}: Fill vs Step")
        ax.set_xlabel("Step Size")
        ax.set_ylabel("Mean Fill Percentage")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    save_figure(fig, "fill_vs_step")
    plt.show()


def plot_runtime_vs_k(df, k_order, step_order):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = df[df["dataset"] == dataset]
        for step in step_order:
            temp = subset[subset["step"] == step]
            grouped = temp.groupby("k")["elapsed_total"].mean().reindex(k_order)
            x = grouped.index.values
            y = grouped.values
            x_smooth, y_smooth = smooth_curve(x, y)
            ax.plot(x_smooth, y_smooth, linewidth=3, alpha=0.9, label=f"step={step}")
            ax.scatter(x, y, s=80, zorder=3)
        ax.set_title(f"{dataset.capitalize()}: Runtime vs k")
        ax.set_xlabel("Buffer Size (k)")
        ax.set_ylabel("Mean Runtime (s)")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    save_figure(fig, "runtime_vs_k")
    plt.show()


def plot_fill_vs_runtime(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"blockout": "o", "kitchen": "s"}
    for dataset in DATASETS:
        subset = df[df["dataset"] == dataset]
        grouped = subset.groupby(["k", "step"]).agg({"fill_percent": "mean", "elapsed_total": "mean"}).reset_index()
        grouped = grouped.sort_values("elapsed_total")
        x = grouped["elapsed_total"].values
        y = grouped["fill_percent"].values
        x_smooth, y_smooth = smooth_curve(x, y)
        ax.plot(x_smooth, y_smooth, linewidth=3, alpha=0.9, label=dataset)
        ax.scatter(x, y, s=100, alpha=0.8, marker=markers[dataset])
    ax.set_title("Fill vs Runtime Tradeoff")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Mean Fill Percentage")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    save_figure(fig, "fill_vs_runtime")
    plt.show()


def plot_convergence(grasp_df, step_order):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = grasp_df[grasp_df["dataset"] == dataset]
        for step in step_order:
            temp = subset[subset["step"] == step]
            if temp.empty:
                continue
            grouped = temp.groupby("grasp_iter")["iter_fill"].mean()
            std = temp.groupby("grasp_iter")["iter_fill"].std()
            x = grouped.index.values
            y = grouped.values
            y_std = std.fillna(0).values
            ax.plot(x, y, marker="o", linewidth=3, label=f"step={step}")
            ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)
        ax.set_title(f"{dataset.capitalize()}: GRASP Convergence")
        ax.set_xlabel("GRASP Iteration")
        ax.set_ylabel("Mean Fill Percentage")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    save_figure(fig, "convergence")
    plt.show()


def plot_objects_vs_step(df, k_order, step_order):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = df[df["dataset"] == dataset]
        for k in k_order:
            temp = subset[subset["k"] == k]
            grouped = temp.groupby("step")["placed"].mean().reindex(step_order)
            x = grouped.index.values
            y = grouped.values
            x_smooth, y_smooth = smooth_curve(x, y)
            ax.plot(x_smooth, y_smooth, linewidth=3, alpha=0.9, label=f"k={k}")
            ax.scatter(x, y, s=80, zorder=3)
        ax.set_title(f"{dataset.capitalize()}: Objects Packed vs Step")
        ax.set_xlabel("Step Size")
        ax.set_ylabel("Mean Objects Packed")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    save_figure(fig, "objects_vs_step")
    plt.show()


def plot_fill_distribution(df, k_order):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = df[df["dataset"] == dataset]
        data_to_plot = []
        labels = []
        for k in k_order:
            temp = subset[subset["k"] == k]
            data_to_plot.append(temp["fill_percent"].values)
            labels.append(f"k={k}")
        ax.boxplot(data_to_plot, tick_labels=labels)
        ax.set_title(f"{dataset.capitalize()}: Fill Distribution")
        ax.set_xlabel("Buffer Size (k)")
        ax.set_ylabel("Fill Percentage")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, "fill_distribution")
    plt.show()


def plot_objects_distribution(df, k_order):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = df[df["dataset"] == dataset]
        data_to_plot = []
        labels = []
        for k in k_order:
            temp = subset[subset["k"] == k]
            data_to_plot.append(temp["placed"].values)
            labels.append(f"k={k}")
        ax.boxplot(data_to_plot, tick_labels=labels)
        ax.set_title(f"{dataset.capitalize()}: Objects Packed Distribution")
        ax.set_xlabel("Buffer Size (k)")
        ax.set_ylabel("Objects Packed")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, "objects_distribution")
    plt.show()


def main():
    ensure_output_dirs()
    df, grasp_df = load_meta_results()

    if df.empty:
        print("No valid data loaded. Revisa la carpeta resultados/.../meta y el contenido JSON.")
        return

    print("\n========================")
    print("DATAFRAME INFO")
    print("========================")
    print(df.head())
    print("\nUnique k values:")
    print(sorted(df["k"].dropna().unique()))
    print("\nUnique step values:")
    print(sorted(df["step"].dropna().unique()))
    print("\nShape:")
    print(df.shape)

    k_order = sorted(df["k"].dropna().unique())
    step_order = sorted(df["step"].dropna().unique())

    raw_tables = save_raw_tables(df)
    summaries = save_summary_tables(df)
    top5_tables = save_top5_tables(df)
    write_excel(raw_tables, summaries, top5_tables)

    plot_fill_vs_step(df, k_order, step_order)
    plot_runtime_vs_k(df, k_order, step_order)
    plot_convergence(grasp_df, step_order)
    plot_objects_vs_step(df, k_order, step_order)
    plot_metric_heatmap(df, "fill_percent", "Mean Fill Percent Heatmap", "fill_heatmap")
    plot_metric_heatmap(df, "elapsed_total", "Mean Runtime Heatmap", "runtime_heatmap")
    plot_placed_heatmap(df)

    print("\nDONE.")


if __name__ == "__main__":
    main()
