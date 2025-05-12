"""
Utility functions for analyzing and visualizing data quality flags
and SIC code distributions.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Any, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# --- Default Configuration Values (if not found in config) ---
DEFAULT_OUTPUT_DIR = "analysis_outputs"
DEFAULT_SIC_OCC1_COL = "sic_ind_occ1"
DEFAULT_SIC_OCC2_COL = "sic_ind_occ2"
TOP_N_HISTOGRAM = 20 # Number of top items to show in SIC code histograms


def plot_sic_code_histogram(
    df: pd.DataFrame,
    column_name: str,
    output_dir: Path,
    top_n: int = TOP_N_HISTOGRAM,
    filename_suffix: str = "",
) -> None:
    """
    Generates and saves a histogram (bar plot) for the value counts of a SIC code column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the SIC code column to analyze.
        output_dir (Path): The directory to save the plot.
        top_n (int): Number of top most frequent items to display.
        filename_suffix (str): Suffix to add to the plot filename.
    """
    if column_name not in df.columns:
        logger.warning(
            "Column '%s' not found in DataFrame. Skipping histogram generation.",
            column_name
        )
        return
    if df[column_name].isnull().all():
        logger.warning(
            "Column '%s' contains all null/NaN values. Skipping histogram generation.",
            column_name
        )
        return

    logger.info("Generating histogram for column: %s", column_name)
    plt.figure(figsize=(12, 8))
    counts = df[column_name].value_counts().nlargest(top_n)

    if counts.empty:
        logger.warning("No data to plot for histogram of column '%s'.", column_name)
        plt.close() # Close the empty figure
        return

    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title(f"Top {top_n} Most Frequent Codes in '{column_name}' (Total Rows: {len(df)})")
    plt.xlabel(f"{column_name} Code")
    plt.ylabel("Frequency (Count)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_filename = f"{column_name.lower()}_distribution{filename_suffix}.png"
    output_path = output_dir / output_filename
    try:
        plt.savefig(output_path)
        logger.info("Histogram saved to %s", output_path)
    except Exception as e:
        logger.exception("Failed to save histogram for %s: %s", column_name, e)
    plt.close()


def plot_boolean_flag_summary(
    df: pd.DataFrame,
    flag_columns: List[str],
    output_dir: Path,
    filename_suffix: str = "",
) -> None:
    """
    Generates and saves a bar plot summarizing counts and percentages of True values
    for specified boolean flag columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        flag_columns (List[str]): A list of boolean column names to analyze.
        output_dir (Path): The directory to save the plot.
        filename_suffix (str): Suffix to add to the plot filename.
    """
    valid_flag_columns = [col for col in flag_columns if col in df.columns]
    if not valid_flag_columns:
        logger.warning("None of the specified flag columns found. Skipping summary plot.")
        return

    logger.info("Generating summary plot for boolean flags: %s", valid_flag_columns)

    true_counts = []
    percentages = []
    total_rows = len(df)

    if total_rows == 0:
        logger.warning("Input DataFrame is empty. Skipping boolean flag summary.")
        return

    for col in valid_flag_columns:
        # Ensure column is boolean or can be safely summed (True=1, False=0)
        # Pandas nullable boolean 'boolean' dtype handles sum correctly (ignores NA)
        if pd.api.types.is_bool_dtype(df[col]) or df[col].dtype.name == 'boolean':
            count = df[col].sum() # True values are summed as 1
            true_counts.append(count)
            percentages.append((count / total_rows) * 100 if total_rows > 0 else 0)
        else:
            logger.warning(
                "Column '%s' is not boolean type. Summing True-like strings if any.", col
            )
            # Attempt to count True-like strings (less robust)
            true_like_count = df[col].astype(str).str.upper().isin(['TRUE', '1']).sum()
            true_counts.append(true_like_count)
            percentages.append((true_like_count / total_rows) * 100 if total_rows > 0 else 0)


    summary_df = pd.DataFrame({
        'Flag': valid_flag_columns,
        'Count_True': true_counts,
        'Percentage_True': percentages
    })

    if summary_df.empty:
        logger.warning("No data to plot for boolean flag summary.")
        return

    # Plotting Counts
    plt.figure(figsize=(12, 7))
    barplot = sns.barplot(x='Flag', y='Count_True', data=summary_df, palette="Blues_d")
    plt.title(f"Count of TRUE Values for Quality Flags (Total Rows: {total_rows})")
    plt.xlabel("Flag Name")
    plt.ylabel("Count of TRUE")
    plt.xticks(rotation=45, ha="right")

    # Add percentage annotations
    for i, p in enumerate(barplot.patches):
        height = p.get_height()
        percentage_val = summary_df.loc[i, 'Percentage_True']
        barplot.text(
            p.get_x() + p.get_width() / 2.0,
            height + 0.01 * total_rows, # Offset text slightly above bar
            f'{height}\n({percentage_val:.1f}%)',
            ha='center',
            va='bottom'
        )

    plt.tight_layout()
    output_filename_counts = f"boolean_flags_summary{filename_suffix}.png"
    output_path_counts = output_dir / output_filename_counts
    try:
        plt.savefig(output_path_counts)
        logger.info("Boolean flags summary plot saved to %s", output_path_counts)
    except Exception as e:
        logger.exception("Failed to save boolean flags summary plot: %s", e)
    plt.close()


def perform_graphical_analysis(
    df: pd.DataFrame,
    config: Optional[dict[str, Any]] = None,
    run_id: Optional[str] = None
) -> None:
    """
    Performs graphical analysis on the DataFrame for SIC codes and quality flags.

    Args:
        df (pd.DataFrame): The DataFrame to analyze (e.g., sic_dataframe_with_flags).
        config (Optional[dict[str, Any]]): Loaded configuration dictionary.
        run_id (Optional[str]): An optional ID to append to filenames for uniqueness.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping graphical analysis.")
        return

    # Determine output directory
    output_dir_str = DEFAULT_OUTPUT_DIR
    project_root = Path.cwd() # Default if config or project_root not easily found
    if config and "paths" in config and "output_dir" in config["paths"]:
        # Assuming config paths are relative to a project root if using AppConfig class
        # For standalone function, assume output_dir is relative to CWD or absolute
        output_dir_str = config["paths"]["output_dir"]
        # If using AppConfig structure, project_root would be config.project_root
        # Here, we'll just use the path as given or relative to CWD
        output_dir = Path(output_dir_str)
    else:
        output_dir = project_root / output_dir_str
        logger.warning(
            "Output directory not found in config. Using default: %s", output_dir
        )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.exception("Could not create output directory %s: %s. Plots will not be saved.", output_dir, e)
        return # Cannot save plots if directory creation fails

    filename_suffix = f"_{run_id}" if run_id else ""

    # Get column names from config or use defaults
    col_names_config = config.get("column_names", {}) if config else {}
    sic_occ1_col = col_names_config.get("gold_sic", DEFAULT_SIC_OCC1_COL) # Using 'gold_sic' from your config
    # Assuming 'sic_ind_occ2' is not explicitly in config, use a default or add it
    sic_occ2_col = col_names_config.get("input_csv_columns_map", {}).get("sic_ind_occ2", DEFAULT_SIC_OCC2_COL)
    # If input_csv_columns is a list, we can't directly map.
    # For simplicity, using defaults if not directly specified.
    # A more robust config would have specific keys for these analysis columns.

    # Analysis for sic_ind_occ1 and sic_ind_occ2
    plot_sic_code_histogram(df, sic_occ1_col, output_dir, filename_suffix=filename_suffix)
    plot_sic_code_histogram(df, sic_occ2_col, output_dir, filename_suffix=filename_suffix)

    # Analysis for Boolean Flag Columns
    boolean_flags_to_analyze = [
        "Single_Answer", "Unambiguous", "Match_5_digits", "Match_2_digits"
    ]
    plot_boolean_flag_summary(df, boolean_flags_to_analyze, output_dir, filename_suffix=filename_suffix)

    logger.info("Graphical analysis complete. Plots saved to %s", output_dir)


# --- Example Usage (if this script were run directly) ---
if __name__ == '__main__':
    # This example assumes you have a config.toml and a way to load your DataFrame
    # For testing, you might create a dummy DataFrame here.

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # 1. Load Configuration (Example - adapt to your actual config loading)
    try:
        from src.config_loader import load_config # Assuming you have this
        app_config = load_config("config.toml") # Path to your config
    except ImportError:
        logger.warning("config_loader not found, using None for config.")
        app_config = None
    except FileNotFoundError:
        logger.warning("config.toml not found, using None for config.")
        app_config = None


    # 2. Create or Load your sic_dataframe_with_flags
    # For demonstration, creating a dummy DataFrame:
    data_dict = {
        'sic_ind_occ1': ['12345', '54321', '12345', '98765', '54321', None, '11111', 'XXXXX'],
        'sic_ind_occ2': ['67890', None, '67890', 'NA', None, '12345', 'NA', '12345'],
        'Single_Answer': pd.Series([False, True, False, True, True, False, True, False], dtype='boolean'),
        'Unambiguous': pd.Series([False, False, False, True, False, pd.NA, True, False], dtype='boolean'),
        'Match_5_digits': pd.Series([True, False, True, True, False, pd.NA, True, False], dtype='boolean'),
        'Match_2_digits': pd.Series([False, False, False, False, False, pd.NA, False, False], dtype='boolean'),
        'Other_Col': list(range(8))
    }
    sample_df = pd.DataFrame(data_dict)

    # 3. Perform analysis
    if not sample_df.empty:
        perform_graphical_analysis(sample_df, config=app_config, run_id="test_run")
    else:
        logger.info("Sample DataFrame is empty, skipping analysis.")

