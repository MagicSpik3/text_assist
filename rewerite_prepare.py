"""Load and Filter utilities.

The .toml configuration file should include:
- The path to the test data file (can be local or a gs:// URI).
- Column names mapping if they differ from defaults.

The utilities allow for the following:
1. Loads the configuration from the .toml file.
2. Loads the test data from local or GCS.
3. Adds data quality flag columns to the DataFrame.
"""

import logging
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import toml

# --- Default Configuration Values ---
DEFAULT_SIC_OCC1_COL = "sic_ind_occ1"
DEFAULT_SIC_OCC2_COL = "sic_ind_occ2"
DEFAULT_SIC_OCC3_COL = "sic_ind_occ3"
SPECIAL_SIC_NOT_CODEABLE = "-9"
SPECIAL_SIC_MULTIPLE_POSSIBLE = "4+"
EXPECTED_SIC_LENGTH = 5
X_COUNT_FOR_MATCH_3 = 2
X_COUNT_FOR_MATCH_2 = 3


def load_config(config_path: str) -> dict[str, Any]:
    """Loads configuration settings from a .toml file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return toml.load(file)


def _safe_zfill(value: Any) -> Any:
    """
    Safely pads a value with leading zeros to 5 digits, handling special cases.
    """
    if pd.isna(value):
        return value
    s_value = str(value)
    if s_value in ["-9", "4+"]:
        return s_value
    try:
        return str(int(float(s_value))).zfill(5)
    except (ValueError, TypeError):
        return s_value


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
    return num_answers


def _extract_sic_division(
    sic_occ1_series: pd.Series,
    not_codeable_flag: pd.Series,
    multiple_possible_flag: pd.Series,
) -> pd.Series:
    """Extracts the first two digits (division) from the sic_ind_occ1 series."""
    sic_division = pd.Series("", index=sic_occ1_series.index, dtype=str)
    starts_with_two_digits = sic_occ1_series.str.match(r"^[0-9]{2}")
    eligible = starts_with_two_digits & ~not_codeable_flag & ~multiple_possible_flag
    sic_division[eligible] = sic_occ1_series[eligible].str[:2]
    return sic_division


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

    required_cols = [col_occ1, col_occ2, col_occ3]
    if not all(col in df_out.columns for col in required_cols):
        missing = set(required_cols) - set(df_out.columns)
        logging.error("Input DataFrame missing columns: %s.", missing)
        return df

    df_out["Not_Codeable"] = df_out[col_occ1] == SPECIAL_SIC_NOT_CODEABLE
    df_out["Four_Or_More"] = df_out[col_occ1] == SPECIAL_SIC_MULTIPLE_POSSIBLE
    df_out["SIC_Division"] = _extract_sic_division(
        df_out[col_occ1], df_out["Not_Codeable"], df_out["Four_Or_More"]
    )
    df_out["num_answers"] = _calculate_num_answers(df_out, col_occ1, col_occ2, col_occ3)
    df_out.loc[df_out["Not_Codeable"], "num_answers"] = 0
    df_out.loc[df_out["Four_Or_More"], "num_answers"] = 4

    s_occ1 = df_out[col_occ1].fillna("").astype(str)
    for name, series in _create_sic_match_flags(s_occ1).items():
        df_out[name] = series

    if "Match_5_digits" in df_out.columns:
        df_out["Unambiguous"] = (df_out["num_answers"] == 1) & df_out["Match_5_digits"]
    else:
        df_out["Unambiguous"] = False

    flag_cols = ["Match_5_digits", "Match_3_digits", "Match_2_digits", "Unambiguous"]
    for col in flag_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype("boolean")

    logging.info("Finished adding data quality flag columns.")
    return df_out


if __name__ == "__main__":
    main_config = load_config("config.toml")
    log_config = main_config.get("logging", {})
    logging.basicConfig(
        level=log_config.get("level", "INFO").upper(),
        format=log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s"),
        filename=log_config.get("file"),
    )

    analysis_filepath = main_config["paths"]["batch_filepath"]
    analysis_csv = main_config["paths"]["analysis_csv"]

    try:
        # --- FIX: Read directly from local path or GCS URI ---
        # Pandas will automatically use gcsfs if the path starts with "gs://"
        logging.info("Loading data from: %s", analysis_filepath)
        sic_dataframe = pd.read_csv(analysis_filepath, delimiter=",", dtype=str)

        # --- NEW: Apply leading zero fix right after loading ---
        logging.info("Applying leading zero padding to SIC columns...")
        sic_cols_to_fix = [DEFAULT_SIC_OCC1_COL, DEFAULT_SIC_OCC2_COL, DEFAULT_SIC_OCC3_COL]
        for col in sic_cols_to_fix:
            if col in sic_dataframe.columns:
                sic_dataframe[col] = sic_dataframe[col].apply(_safe_zfill)

        # Add quality flags to the cleaned data
        sic_dataframe_with_flags = add_data_quality_flags(sic_dataframe, main_config)

        print("\nDataFrame Head with Quality Flags:")
        print(sic_dataframe_with_flags.head())

        # --- FIX: Only check existence for local paths, not GCS paths ---
        if not analysis_csv.startswith("gs://"):
            output_dir = os.path.dirname(analysis_csv)
            os.makedirs(output_dir, exist_ok=True)

        logging.info("Saving DataFrame with flags to: %s", analysis_csv)
        sic_dataframe_with_flags.to_csv(analysis_csv, index=False)

    except FileNotFoundError:
        logging.error("Input file not found at: %s", analysis_filepath)
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)

