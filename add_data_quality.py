"""Load and Filter utilities.

The .toml configuration file should include:
- The path to the test data file.
- Column names mapping if they differ from defaults.

The utilities allow for the following:
1. Loads the configuration from the .toml file.
2. Loads the test data.
3. Adds data quality flag columns to the DataFrame.
"""

import logging
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import toml

# --- Default Configuration Values (if not found in config) ---
DEFAULT_OUTPUT_DIR = "analysis_outputs"
DEFAULT_SIC_OCC1_COL = "sic_ind_occ1"
DEFAULT_SIC_OCC2_COL = "sic_ind_occ2"
DEFAULT_SIC_OCC3_COL = "sic_ind_occ3"

SPECIAL_SIC_NOT_CODEABLE = "-9"
SPECIAL_SIC_MULTIPLE_POSSIBLE = "4+"

# --- Constants for Data Quality ---
EXPECTED_SIC_LENGTH = 5
X_COUNT_FOR_MATCH_3 = 2
X_COUNT_FOR_MATCH_2 = 3


# Load the config:
def load_config(config_path):
    """Loads configuration settings from a .toml file.

    Args:
        config_path (str): The path to the .toml configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    with open(config_path, encoding="utf-8") as file:
        configuration = toml.load(file)
    return configuration


# --- Helper Function for SIC Code Matching ---
def _create_sic_match_flags(sic_series: pd.Series) -> dict[str, pd.Series]:
    """Calculates various SIC code format match flags for a given Series."""
    flags = {}
    flags["Match_5_digits"] = sic_series.str.match(r"^[0-9]{5}$", na=False)
    is_len_expected = sic_series.str.len() == EXPECTED_SIC_LENGTH
    x_count = sic_series.str.count("x", re.I)
    only_digits_or_x = sic_series.str.match(r"^[0-9xX]*$", na=False)
    non_x_part = sic_series.str.replace("x", "", case=False)
    are_non_x_digits = (non_x_part != "") & non_x_part.str.match(r"^[0-9]*$", na=False)
    base_partial_check = is_len_expected & only_digits_or_x & are_non_x_digits
    flags["Match_3_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_3)
    flags["Match_2_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_2)
    return flags


def _calculate_num_answers(
    df: pd.DataFrame, col_occ1: str, col_occ2: str, col_occ3: str
) -> pd.Series:
    """Calculates the number of provided answers in SIC occurrence columns."""
    num_answers = pd.Series(0, index=df.index, dtype="int")
    for col_name in [col_occ1, col_occ2, col_occ3]:
        if col_name in df.columns:
            is_valid_entry = (
                ~df[col_name].isna()
                & (df[col_name].astype(str).str.strip() != "")
                & (df[col_name].astype(str).str.upper() != "NA")
            )
            num_answers += is_valid_entry.astype(int)
        else:
            logging.warning(
                "Column '%s' not found for num_answers calculation. It will be ignored.",
                col_name,
            )
    return num_answers


def _extract_sic_division(
    sic_occ1_series: pd.Series,
    not_codeable_flag: pd.Series,
    multiple_possible_flag: pd.Series,
) -> pd.Series:
    """Extracts the first two digits (division) from the sic_ind_occ1 series."""
    logging.debug("Extracting SIC division (first two digits) from sic_ind_occ1.")
    sic_division = pd.Series("", index=sic_occ1_series.index, dtype=str)
    starts_with_two_digits = sic_occ1_series.str.match(r"^[0-9]{2}")
    eligible_for_extraction = (
        starts_with_two_digits & ~not_codeable_flag & ~multiple_possible_flag
    )
    sic_division[eligible_for_extraction] = sic_occ1_series[
        eligible_for_extraction
    ].str[:2]
    logging.debug("Finished extracting SIC division.")
    return sic_division


def _safe_zfill(value: Any) -> Any:
    """
    Safely pads a value with leading zeros to 5 digits.

    - Ignores NaNs.
    - Returns non-numeric strings (like '1234x') as-is.
    - Pads numeric-like strings ('1234') to '01234'.
    """
    if pd.isna(value):
        return value
    try:
        # Attempt to convert to float then int to handle "1234.0" cases
        return str(int(float(value))).zfill(5)
    except (ValueError, TypeError):
        # If conversion fails, it's a non-numeric string like '1234x'
        return value


# --- Data Quality Flagging ---
def add_data_quality_flags(
    df: pd.DataFrame, loaded_config: Optional[dict[str, Any]] = None
) -> pd.DataFrame:
    """Adds data quality flag columns to the DataFrame based on SIC/SOC codes."""
    logging.info("Adding data quality flag columns...")
    df_out = df.copy()

    col_names = loaded_config.get("column_names", {}) if loaded_config else {}
    col_occ1 = col_names.get("sic_ind_occ1", DEFAULT_SIC_OCC1_COL)
    col_occ2 = col_names.get("sic_ind_occ2", DEFAULT_SIC_OCC2_COL)
    col_occ3 = col_names.get("sic_ind_occ3", DEFAULT_SIC_OCC3_COL)

    required_input_cols = [col_occ1, col_occ2, col_occ3]
    if not all(col in df_out.columns for col in required_input_cols):
        missing = set(required_input_cols) - set(df_out.columns)
        logging.error(
            "Input DataFrame missing columns: %s. Skipping flag generation.",
            missing,
        )
        return df

    # --- NEW: Clean and format SIC codes first ---
    logging.info("Applying leading zero padding to SIC columns...")
    for col in required_input_cols:
        df_out[col] = df_out[col].apply(_safe_zfill)

    # --- 1. Special SIC Code Flags for col_occ1 ---
    df_out["Not_Codeable"] = df_out[col_occ1] == SPECIAL_SIC_NOT_CODEABLE
    df_out["Four_Or_More"] = df_out[col_occ1] == SPECIAL_SIC_MULTIPLE_POSSIBLE

    df_out["SIC_Division"] = _extract_sic_division(
        df_out[col_occ1], df_out["Not_Codeable"], df_out["Four_Or_More"]
    )

    # --- 2. Number of Answers ---
    df_out["num_answers"] = _calculate_num_answers(df_out, col_occ1, col_occ2, col_occ3)
    df_out.loc[df_out["Not_Codeable"], "num_answers"] = 0
    df_out.loc[df_out["Four_Or_More"], "num_answers"] = 4

    # --- 3. Digit/Character Match Flags for col_occ1 ---
    s_occ1 = df_out[col_occ1].fillna("").astype(str)
    match_flags_dict = _create_sic_match_flags(s_occ1)
    for flag_name, flag_series in match_flags_dict.items():
        df_out[flag_name] = flag_series

    # --- 4. Unambiguous Flag ---
    if "Match_5_digits" in df_out.columns:
        df_out["Unambiguous"] = (df_out["num_answers"] == 1) & df_out["Match_5_digits"]
    else:
        df_out["Unambiguous"] = False
        logging.warning("'Match_5_digits' not found for 'Unambiguous' flag calculation.")

    # --- 5. Convert to Pandas Nullable Boolean Type ---
    flag_cols_list = ["Match_5_digits", "Match_3_digits", "Match_2_digits", "Unambiguous"]
    for flag_col_name in flag_cols_list:
        if flag_col_name in df_out.columns:
            try:
                df_out[flag_col_name] = df_out[flag_col_name].astype("boolean")
            except (TypeError, ValueError) as e:
                logging.warning("Could not convert column '%s' to boolean: %s", flag_col_name, e)

    logging.info("Finished adding data quality flag columns.")
    return df_out


if __name__ == "__main__":
    # --- Main execution block ---
    main_config = load_config("config.toml")
    log_config = main_config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log_format = log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")
    log_file = log_config.get("file")

    logging.basicConfig(level=log_level, format=log_format)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.getLogger().addHandler(logging.FileHandler(log_file))

    analysis_filepath = main_config["paths"]["batch_filepath"]
    analysis_csv = main_config["paths"]["analysis_csv"]
    output_dir = os.path.dirname(analysis_csv)

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Required output directory not found: {output_dir}")

    sic_dataframe = pd.read_csv(analysis_filepath, delimiter=",", dtype=str)
    sic_dataframe_with_flags = add_data_quality_flags(sic_dataframe, main_config)

    print("\nDataFrame Head with Quality Flags:")
    print(sic_dataframe_with_flags.head())

    sic_dataframe_with_flags.to_csv(analysis_csv, index=False)
    logging.info("Saved DataFrame with flags to %s", analysis_csv)
