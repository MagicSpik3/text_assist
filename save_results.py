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
# # Evaluation Matrix Runner
# 
# This notebook runs a matrix of test scenarios, collates the numerical results,
# and saves them to a summary CSV file.

# %%
# --- Imports and Setup ---
from pathlib import Path
from typing import TypedDict, Dict, Any, List
from datetime import datetime
import pandas as pd
from IPython.display import Markdown, display

# Make sure your classes are importable
from survey_assist_utils.evaluation.coder_alignment import ColumnConfig, LabelAccuracy
from survey_assist_utils.logging import get_logger

logger = get_logger(__name__)

# %%
# --- Step 1: Create a function that CALCULATES the results ---

def calculate_analysis_metrics(
    analyzer: LabelAccuracy, config: ColumnConfig
) -> Dict[str, Any]:
    """
    Runs a suite of analyses and returns the results in a dictionary.
    This function contains no display logic.

    Args:
        analyzer: An initialized LabelAccuracy instance.
        config: The configuration object used for the analysis.

    Returns:
        A dictionary containing all calculated metrics.
    """
    full_acc = analyzer.get_accuracy(match_type="full")
    digit_acc = analyzer.get_accuracy(match_type="2-digit")
    coverage = analyzer.get_coverage()
    
    # Get the detailed summary stats dictionary
    summary_stats = analyzer.get_summary_stats()

    # Combine all results into a single dictionary for easy export
    results = {
        "accuracy_full_match": f"{full_acc:.1f}",
        "accuracy_2_digit_match": f"{digit_acc:.1f}",
        "overall_coverage": f"{coverage:.1f}",
        **summary_stats  # Unpack the summary stats dict into this one
    }
    return results

# --- Step 2: Create a function that DISPLAYS the results ---

def display_analysis_results(
    file_path: Path, test_description: str, analyzer: LabelAccuracy
):
    """
    Displays the results of an analysis in a structured Jupyter format.
    This function is purely for visual output.
    """
    # Header
    display(Markdown(f"--- \n## 📊 Analysis for: `{file_path.name}`"))
    display(Markdown(f"**Test Scenario:** {test_description}"))
    display(Markdown(f"**Shape:** {analyzer.df.shape}"))

    # Key Metrics Table
    display(Markdown("### Key Accuracy Metrics"))
    full_acc = analyzer.get_accuracy(match_type="full")
    digit_acc = analyzer.get_accuracy(match_type="2-digit")
    key_metrics_df = pd.Series({
        "Overall Accuracy (Full Match)": f"{full_acc:.1f}%",
        "Overall Accuracy (2-Digit Match)": f"{digit_acc:.1f}%",
        "Overall Coverage": f"{analyzer.get_coverage():.1f}%",
    }).to_frame("Value")
    display(key_metrics_df)
    
    # Visualizations
    display(Markdown("### Visualizations"))
    analyzer.plot_threshold_curves()
    analyzer.plot_confusion_heatmap(
        human_code_col=analyzer.config.clerical_label_cols[0],
        llm_code_col=analyzer.config.model_label_cols[0],
        top_n=10,
        exclude_patterns=["x", "-9"],
    )

# %%
# --- Step 3: Define the Test Matrix and Configuration ---

class TestCase(TypedDict):
    Test: str
    CCs: list[int]
    LLMs: list[int]
    Unambiguous: bool

file_directory = Path("/home/user/survey-assist-utils/data/evaluation_data/")
file_to_test = "DSC_Rep_Sample_test_end_to_end_20250620_142222_output.csv"
full_file_path = file_directory / file_to_test

model_label_cols = [f"candidate_{i}_sic_code" for i in range(1, 6)]
model_score_cols = [f"candidate_{i}_likelihood" for i in range(1, 6)]
clerical_label_cols = [f"sic_ind_occ{i}" for i in range(1, 4)]

test_cases: List[TestCase] = [
    {"Test": "Top SA vs Top CC, All Data", "CCs": [1], "LLMs": [1], "Unambiguous": False},
    {"Test": "Top SA vs Top CC, Unambiguous", "CCs": [1], "LLMs": [1], "Unambiguous": True},
    {"Test": "Any of 5 SA vs Any of 3 CC, All Data", "CCs": [3], "LLMs": [5], "Unambiguous": False},
    {"Test": "Any of 5 SA vs Any of 3 CC, Unambiguous", "CCs": [3], "LLMs": [5], "Unambiguous": True},
    {"Test": "Any of 5 SA vs Top CC, All Data", "CCs": [1], "LLMs": [5], "Unambiguous": False},
    {"Test": "Any of 5 SA vs Top CC, Unambiguous", "CCs": [1], "LLMs": [5], "Unambiguous": True},
]

# --- Step 4: Run the Matrix, Collect Results, and Save to CSV ---

# Load the primary data file once to avoid re-reading it in the loop
try:
    main_df = pd.read_csv(full_file_path, dtype=str)
    logger.info(f"Successfully loaded main data file with shape: {main_df.shape}")
except FileNotFoundError:
    logger.error(f"Main data file not found: {full_file_path}")
    main_df = None

all_results = []

if main_df is not None:
    for case in test_cases:
        for cc_count in case["CCs"]:
            for llm_count in case["LLMs"]:
                # Create the specific configuration for this test run
                config = ColumnConfig(
                    model_label_cols=model_label_cols[:llm_count],
                    model_score_cols=model_score_cols[:llm_count],
                    clerical_label_cols=clerical_label_cols[:cc_count],
                    id_col="unique_id",
                    filter_unambiguous=case["Unambiguous"],
                )
                
                # Initialize the analyzer with the data and config
                analyzer = LabelAccuracy(df=main_df.copy(), column_config=config)
                
                # ---- A. Calculate the metrics ----
                test_results = calculate_analysis_metrics(analyzer, config)
                
                # ---- B. Display the visuals (optional) ----
                display_analysis_results(
                    file_path=full_file_path,
                    test_description=case["Test"],
                    analyzer=analyzer
                )

                # ---- C. Collate the results for the final CSV ----
                # Add the test case description to the results dictionary
                record = {"test_scenario": case["Test"], **test_results}
                all_results.append(record)

# --- Step 5: Save the Collated Results ---

if all_results:
    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Create a timestamped filename for the output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"evaluation_summary_{timestamp}.csv"
    output_path = file_directory / output_filename
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Successfully saved evaluation summary to: {output_path}")
    display(Markdown("### 📄 Final Collated Results"))
    display(results_df)
