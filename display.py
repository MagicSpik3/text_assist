from pathlib import Path
from IPython.display import Markdown, display
import pandas as pd

# Assume your LabelAccuracy class and ColumnConfig are imported
# from survey_assist_utils.evaluation.coder_alignment import ColumnConfig, LabelAccuracy

def display_analysis_results(
    file_path: Path, test_description: str, analyzer: LabelAccuracy
):
    """
    Displays the results of an analysis in a structured Jupyter format.
    This version correctly handles the dictionary returned by get_accuracy.

    Args:
        file_path (Path): The full path to the test CSV file.
        test_description (str): The description of the evaluation being run.
        analyzer: An initialized LabelAccuracy instance.
    """
    # --- 1. Header and Data Loading Info ---
    display(Markdown(f"--- \n## 📊 Analysis for: `{file_path.name}`"))
    display(Markdown(f"**Test Scenario:** {test_description}"))
    display(Markdown(f"**Shape:** {analyzer.df.shape}"))

    # --- 2. Key Metrics (The Updated Section) ---
    display(Markdown("### Key Accuracy Metrics"))
    
    # Call the methods to get the full statistics dictionaries
    full_acc_stats = analyzer.get_accuracy(match_type="full")
    digit_acc_stats = analyzer.get_accuracy(match_type="2-digit")

    # Display key metrics, now including the raw counts for verification
    key_metrics_df = pd.Series({
        "Overall Accuracy (Full Match)": f"{full_acc_stats['accuracy_percent']:.1f}%",
        "Full Match Count": f"{full_acc_stats['matches']} / {full_acc_stats['total_considered']}",
        "Overall Accuracy (2-Digit Match)": f"{digit_acc_stats['accuracy_percent']:.1f}%",
        "2-Digit Match Count": f"{digit_acc_stats['matches']} / {digit_acc_stats['total_considered']}",
        "Overall Coverage": f"{analyzer.get_coverage():.1f}%",
    }).to_frame("Value")
    display(key_metrics_df)

    # --- 3. Threshold Statistics (Unchanged) ---
    display(Markdown("### Accuracy/Coverage vs. Threshold"))
    threshold_stats = analyzer.get_threshold_stats()
    display(threshold_stats.head())

    # --- 4. Plots (Unchanged) ---
    display(Markdown("### Visualizations"))
    analyzer.plot_threshold_curves()
    analyzer.plot_confusion_heatmap(
        human_code_col=analyzer.config.clerical_label_cols[0],
        llm_code_col=analyzer.config.model_label_cols[0],
        top_n=10,
        exclude_patterns=["x", "-9"],
    )

    # --- 5. Summary Statistics (Unchanged) ---
    display(Markdown("### Summary Statistics Dictionary"))
    summary_stats = analyzer.get_summary_stats()
    display(pd.Series(summary_stats, name="Value").to_frame())
