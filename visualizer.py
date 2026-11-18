"""
Generates comprehensive plots and tables for the thesis.
Saves to results/figures/ and results/tables/
"""
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Setup styling
sns.set_theme(style="whitegrid")
FIG_DIR = "results/figures"
TAB_DIR = "results/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

def load_data():
    with open("results/full_results_llama.json", "r", encoding="utf-8") as f:
        return json.load(f)

def generate_tables(data):
    """Generates CSV tables for the thesis."""
    # Table 1: Overall Metrics Comparison
    metrics_df = pd.DataFrame({
        "Metric": ["Faithfulness", "Answer Relevancy", "Context Precision", "Hallucination Rate"],
        "Baseline": [
            data["baseline_metrics"]["faithfulness"],
            data["baseline_metrics"]["answer_relevancy"],
            data["baseline_metrics"]["context_precision"],
            data["baseline_metrics"]["hallucination_rate"]
        ],
        "GCR": [
            data["gcr_metrics"]["faithfulness"],
            data["gcr_metrics"]["answer_relevancy"],
            data["gcr_metrics"]["context_precision"],
            data["gcr_metrics"]["hallucination_rate"]
        ]
    })
    metrics_df["Delta"] = metrics_df["GCR"] - metrics_df["Baseline"]
    metrics_df["Delta %"] = (metrics_df["Delta"] / metrics_df["Baseline"] * 100).round(1)
    metrics_df.to_csv(f"{TAB_DIR}/table1_overall_metrics.csv", index=False)
    print("Saved table1_overall_metrics.csv")

    # Table 2: GCR Component Breakdown
    gcr_meta = data.get("gcr_metadata", [])
    gcr_df = pd.DataFrame([
        {
            "Question_ID": i,
            "GCR": m.get("gcr"),
            "Se": m.get("gcr_se"),
            "Sc": m.get("gcr_sc"),
            "Ph": m.get("gcr_ph"),
            "Rejected": m.get("rejected", False)
        }
        for i, m in enumerate(gcr_meta) if isinstance(m, dict)
    ])
    gcr_df.to_csv(f"{TAB_DIR}/table2_gcr_breakdown.csv", index=False)
    print("Saved table2_gcr_breakdown.csv")

    # Table 3: Timing and Rejection Stats
    stats_df = pd.DataFrame([{
        "Baseline Time (s)": data.get("baseline_time", 0),
        "GCR Time (s)": data.get("gcr_time", 0),
        "Total Queries": len(gcr_meta),
        "Rejected Queries": data.get("gcr_rejected", 0),
        "Rejection Rate (%)": round(data.get("gcr_rejected", 0) / len(gcr_meta) * 100, 1) if gcr_meta else 0
    }])
    stats_df.to_csv(f"{TAB_DIR}/table3_timing_stats.csv", index=False)
    print("Saved table3_timing_stats.csv")

def generate_figures(data):
    """Generates plots for the thesis."""
    
    # Figure 1: Bar Chart - Baseline vs GCR Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["Faithfulness", "Answer Relevancy", "Context Precision", "Hallucination Rate"]
    baseline = [
        data["baseline_metrics"]["faithfulness"],
        data["baseline_metrics"]["answer_relevancy"],
        data["baseline_metrics"]["context_precision"],
        data["baseline_metrics"]["hallucination_rate"]
    ]
    gcr = [
        data["gcr_metrics"]["faithfulness"],
        data["gcr_metrics"]["answer_relevancy"],
        data["gcr_metrics"]["context_precision"],
        data["gcr_metrics"]["hallucination_rate"]
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, baseline, width, label='Baseline (Vector RAG)', color='#1f77b4')
    ax.bar(x + width/2, gcr, width, label='GCR (GraphRAG)', color='#ff7f0e')
    ax.set_ylabel('Score')
    ax.set_title('RAGAS Metrics: Baseline vs GCR')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig1_metrics_comparison.png", dpi=300)
    plt.close()

    # Figure 2: Scatter - Se vs Sc colored by GCR
    gcr_meta = [m for m in data.get("gcr_metadata", []) if isinstance(m, dict) and m.get("gcr_se") is not None]
    if gcr_meta:
        df = pd.DataFrame([{
            "Se": m.get("gcr_se"),
            "Sc": m.get("gcr_sc"),
            "GCR": m.get("gcr"),
            "Rejected": "Rejected" if m.get("rejected") else "Accepted"
        } for m in gcr_meta])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(data=df, x="Se", y="Sc", hue="Rejected", 
                        palette={"Accepted": "#2ca02c", "Rejected": "#d62728"}, 
                        s=100, alpha=0.7, edgecolor="white")
        ax.axhline(0.2, color="gray", linestyle="--", alpha=0.5, label="Sc=0.2 threshold")
        ax.set_xlabel("Entity Overlap (Se)")
        ax.set_ylabel("Structural Connectivity (Sc)")
        ax.set_title("GCR Components: Se vs Sc")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/fig2_se_vs_sc.png", dpi=300)
        plt.close()

    # Figure 3: Histogram - GCR Distribution
    if gcr_meta:
        gcr_values = [m.get("gcr") for m in gcr_meta if m.get("gcr") is not None]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(gcr_values, bins=20, kde=True, color="purple", ax=ax)
        ax.axvline(0.15, color="red", linestyle="--", label="Threshold (0.15)")
        ax.set_xlabel("GCR Score")
        ax.set_ylabel("Number of Queries")
        ax.set_title("Distribution of GCR Scores")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/fig3_gcr_distribution.png", dpi=300)
        plt.close()

    # Figure 4: Bar chart — Hallucination Rate
    fig, ax = plt.subplots(figsize=(8, 6))
    
    baseline_hr = data["baseline_metrics"]["hallucination_rate"]
    gcr_hr = data["gcr_metrics"]["hallucination_rate"]
    
    bars = ax.bar(
        ["Baseline", "GCR"],
        [baseline_hr, gcr_hr],
        color=["#1f77b4", "#ff7f0e"],
        edgecolor="white",
        linewidth=1.5,
    )
    
    for bar, v in zip(bars, [baseline_hr, gcr_hr]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Hallucination Rate: Baseline vs GCR", fontsize=14)
    ax.set_ylim(0, max(baseline_hr, gcr_hr) * 1.2)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig4_hallucination_bar.png", dpi=300)
    plt.close()
    print(f"  ✓ {FIG_DIR}/fig4_hallucination_bar.png")

    # Figure 5: Pie Chart - Rejection Rate
    if gcr_meta:
        rejected = data.get("gcr_rejected", 0)
        total = len(gcr_meta)
        accepted = total - rejected
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie([accepted, rejected], labels=["Accepted", "Rejected"], 
               colors=["#2ca02c", "#d62728"], autopct='%1.1f%%', startangle=90)
        ax.set_title(f"GCR Context Acceptance (Total: {total})")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/fig5_rejection_rate.png", dpi=300)
        plt.close()

    print("All figures saved to results/figures/")

def generate_sensitivity_plots():
    """Generates plots for the weight sensitivity analysis."""
    path = "results/gcr_grid_search.csv"
    if not os.path.exists(path):
        print("Grid search results not found. Run gcr_sensitivity.py first.")
        return

    df = pd.read_csv(path)
    
    # Filter for a specific threshold to make the heatmap readable
    # We'll look at threshold = 0.15 (the default)
    df_thresh = df[df["threshold"] == 0.15].copy()

    # Figure 6: Heatmap of Acceptance Rate (Alpha vs Beta)
    pivot_accept = df_thresh.pivot_table(
        index="beta", columns="alpha", values="acceptance_rate", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_accept, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    ax.set_title("Acceptance Rate: Alpha (Se) vs Beta (Sc) [Threshold=0.15]")
    ax.set_xlabel("Alpha (Weight of Se)")
    ax.set_ylabel("Beta (Weight of Sc)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig6_heatmap_acceptance.png", dpi=300)
    plt.close()

    # Figure 7: Line Plot - Impact of Gamma on Average GCR
    # Average across all alpha/beta for a fixed threshold
    gamma_impact = df[df["threshold"] == 0.15].groupby("gamma")["avg_gcr"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=gamma_impact, x="gamma", y="avg_gcr", marker="o", ax=ax, color="red")
    ax.set_title("Impact of Gamma (Hub Penalty) on Average GCR [Threshold=0.15]")
    ax.set_xlabel("Gamma (Weight of Ph)")
    ax.set_ylabel("Average GCR Score")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig7_gamma_impact.png", dpi=300)
    plt.close()

    # Figure 8: Multi-line plot - Threshold vs Acceptance Rate for different Alphas
    # Fix Beta=0.4, Gamma=0.2
    df_fixed = df[(df["beta"] == 0.4) & (df["gamma"] == 0.2)].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df_fixed, x="threshold", y="acceptance_rate", hue="alpha", marker="o", ax=ax)
    ax.set_title("Acceptance Rate vs Threshold (Beta=0.4, Gamma=0.2)")
    ax.set_xlabel("GCR Threshold")
    ax.set_ylabel("Acceptance Rate")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig8_threshold_sensitivity.png", dpi=300)
    plt.close()

    print("Sensitivity plots saved to results/figures/")

# Update the main block of visualizer.py to call this new function
if __name__ == "__main__":
    if not os.path.exists("results/full_results.json"):
        print("Error: Run the experiment first.")
    else:
        data = load_data()
        generate_tables(data)
        generate_figures(data)
        generate_sensitivity_plots() # <--- ADD THIS LINE
        print("Done! Check results/figures/ and results/tables/")