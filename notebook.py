# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: survey-assist-utils-PWI-TvqZ-py3.12
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analysis Runner for Unit Test Datasets
# 
# This notebook runs a series of evaluation metrics on a list of test CSV files and displays the results in a structured format.

# %%
# --- Imports and Setup ---
from pathlib import Path
import pandas as pd
from IPython.display import display, Markdown

# Make sure your classes are importable from their location
from survey_assist_utils.evaluation.coder_alignment import ColumnConfig, LabelAccuracy

# %%
def run_and_display_analysis(file_path: Path, config: ColumnConfig):
    """
    Loads a single test file, runs a full suite of analyses, and displays
    the results in a structured and readable format in a Jupyter notebook.

    Args:
        file_path (Path): The full path to the test CSV file.
        config (ColumnConfig): The configuration object defining column names.
    """
    
    # --- 1. Header and Data Loading ---
    # Use Markdown for a clean, bold header for each file's section
    display(Markdown(f"--- \n## 📊 Analysis for: `{file_path.name}`"))
    
    try:
        process_data = pd.read_csv(file_path, dtype=str)
        logging.info("Successfully loaded data with shape: %s", process_data.shape)
        display(Markdown(f"**Shape:** {process_data.shape}"))
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        return

    analyzer = LabelAccuracy(df=process_data, column_config=config)
    
    # --- 2. Key Metrics ---
    display(Markdown("### Key Accuracy Metrics"))
    full_acc = analyzer.get_accuracy(match_type='full')
    digit_acc = analyzer.get_accuracy(match_type='2-digit')
    
    # Display key metrics in a small table for clarity
    key_metrics = pd.Series({
        "Overall Accuracy (Full Match)": f"{full_acc:.1f}%",
        "Overall Accuracy (2-Digit Match)": f"{digit_acc:.1f}%",
        "Overall Coverage": f"{analyzer.get_coverage():.1f}%"
    }).to_frame("Value")
    display(key_metrics)
    
    # --- 3. Threshold Statistics ---
    display(Markdown("### Accuracy/Coverage vs. Threshold"))
    threshold_stats = analyzer.get_threshold_stats()
    display(threshold_stats.head()) # Display the head of the stats table
    
    # --- 4. Plots ---
    display(Markdown("### Visualizations"))
    analyzer.plot_threshold_curves()
    
    # Set up columns for the confusion matrix from the config
    human_code_col = config.clerical_label_cols[0]
    llm_code_col = config.model_label_cols[0]
    
    analyzer.plot_confusion_heatmap(
        human_code_col=human_code_col,
        llm_code_col=llm_code_col,
        top_n=10,
        exclude_patterns=["x", "-9"],
    )
    
    # --- 5. Summary Statistics ---
    display(Markdown("### Summary Statistics Dictionary"))
    summary_stats = analyzer.get_summary_stats()
    # Convert the dictionary to a pandas Series for a nice table display
    display(pd.Series(summary_stats, name="Value").to_frame())


# %%
# --- Main Execution Loop ---

# Set up the configuration that will be used for all test files
col_config = ColumnConfig(
    clerical_label_cols=["CC_1", "CC_2", "CC_3"],
    model_label_cols=["SA_1", "SA_2", "SA_3", "SA_4", "SA_5"],
    model_score_cols=[
        "SA_score_1", "SA_score_2", "SA_score_3", "SA_score_4", "SA_score_5",
    ],
    id_col="unique_id",
)

# Define the directory and the list of files to process
test_directory = Path("/home/user/survey-assist-utils/data/artificial_data")
file_test_list = [
    "unit_test_confidence.csv",
    "unit_test_coverage.csv",
    "unit_test_digits_accuracy.csv",
    "unit_test_label_accuracy.csv",
    "unit_test_heat_map.csv",
]

# Loop through each file and run the complete analysis
for filename in file_test_list:
    full_path = test_directory / filename
    run_and_display_analysis(file_path=full_path, config=col_config)
