"""Load and Filter utilities.

The .toml configuration file should include:
- The path to the gold standard data file.
- Column names mapping if they differ from defaults.

The utilities allow for the following:
1. Loads the configuration from the .toml file.
2. Loads the gold standard data.
3. Adds data quality flag columns to the DataFrame.
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Assuming config_loader.py is in the same src directory
# from src.config_loader import load_config # Make sure this import works in your env

logger = logging.getLogger(__name__)

# --- Constants for Data Quality ---
EXPECTED_SIC_LENGTH = 5
# X_COUNT_FOR_MATCH_4 = 1 # Removed as per request
X_COUNT_FOR_MATCH_3 = 2
X_COUNT_FOR_MATCH_2 = 3
# X_COUNT_FOR_MATCH_1 = 4 # Removed as per request

SPECIAL_SIC_NOT_CODEABLE = "-9"
SPECIAL_SIC_MULTIPLE_POSSIBLE = "+4"


def _create_sic_match_flags(sic_series: pd.Series) -> dict[str, pd.Series]:
    """Calculates various SIC code format match flags for a given Series.

    Args:
        sic_series (pd.Series): A pandas Series containing SIC codes as strings.
                                Missing values should be pre-filled (e.g., with '').

    Returns:
        dict[str, pd.Series]: A dictionary where keys are flag names
                                (e.g., "Match_5_digits") and values are
                                boolean pandas Series.
    """
    flags = {}

    # Match 5 digits: ^[0-9]{5}$
    # This flag is also true if sic_series contains SPECIAL_SIC_NOT_CODEABLE or
    # SPECIAL_SIC_MULTIPLE_POSSIBLE if they are 5 digits.
    # This will be handled by dedicated flags later.
    flags["Match_5_digits"] = sic_series.str.match(r"^[0-9]{5}$", na=False)

    # For partial matches (N digits + X 'x's)
    is_len_expected = sic_series.str.len() == EXPECTED_SIC_LENGTH
    x_count = sic_series.str.count("x", re.I)  # Count 'x' case-insensitively
    only_digits_or_x = sic_series.str.match(r"^[0-9xX]*$", na=False)
    non_x_part = sic_series.str.replace("x", "", case=False)
    # Ensure non_x_part is not empty before checking if it's all digits
    are_non_x_digits = (non_x_part != "") & non_x_part.str.match(
        r"^[0-9]*$", na=False
    )

    base_partial_check = is_len_expected & only_digits_or_x & are_non_x_digits

    # flags["Match_4_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_4) # Removed
    flags["Match_3_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_3)
    flags["Match_2_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_2)
    # flags["Match_1_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_1) # Removed

    return flags


def _calculate_num_answers(
    df: pd.DataFrame, col_occ1: str, col_occ2: str, col_occ3: str
) -> pd.Series:
    """Calculates the number of provided answers in SIC occurrence columns.

    An answer is considered provided if it's not NaN, not an empty string,
    and not "NA" (case-insensitive).

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_occ1 (str): Name of the primary SIC code column.
        col_occ2 (str): Name of the secondary SIC code column.
        col_occ3 (str): Name of the tertiary SIC code column.

    Returns:
        pd.Series: A pandas Series of integers representing the count of answers (0 to 3).
    """
    num_answers = pd.Series(0, index=df.index, dtype="int")
    for col_name in [col_occ1, col_occ2, col_occ3]:
        if col_name in df.columns:
            # An entry is considered valid if it's not NaN, not empty string, and not 'NA'
            is_valid_entry = (
                ~df[col_name].isna()
                & (df[col_name].astype(str).str.strip() != "")
                & (df[col_name].astype(str).str.upper() != "NA")
            )
            num_answers += is_valid_entry.astype(int)
        else:
            logger.warning(
                "Column '%s' not found for num_answers calculation. It will be ignored.",
                col_name,
            )
    return num_answers


# pylint: disable=too-many-locals, too-many-statements
def add_data_quality_flags(
    df: pd.DataFrame, loaded_config: Optional[dict[str, Any]] = None
) -> pd.DataFrame:
    """Adds data quality flag columns to the DataFrame based on SIC/SOC codes.

    Args:
        df (pd.DataFrame): The input DataFrame (typically loaded by read_sic_data).
        loaded_config (Optional[dict]): Loaded configuration dictionary to get column names.

    Returns:
        pd.DataFrame: The DataFrame with added quality flag columns.
                      Returns original DataFrame if essential columns are missing.
    """
    logger.info("Adding data quality flag columns...")
    df_out = df.copy()

    col_names_conf = loaded_config.get("column_names", {}) if loaded_config else {}
    col_occ1 = col_names_conf.get("sic_ind_occ1", "sic_ind_occ1")
    col_occ2 = col_names_conf.get("sic_ind_occ2", "sic_ind_occ2")
    col_occ3 = col_names_conf.get("sic_ind_occ3", "sic_ind_occ3")

    required_input_cols = [col_occ1, col_occ2, col_occ3]
    if not all(col_name in df_out.columns for col_name in required_input_cols):
        missing_cols = set(required_input_cols) - set(df_out.columns)
        logger.error(
            "Input DataFrame missing columns for quality flags: %s. Skipping flag generation.",
            missing_cols,
        )
        return df

    # --- 0. Prepare sic_ind_occ1 series (used by multiple flags) ---
    # Fill NaN with empty string for string operations, then convert to string type
    s_occ1 = df_out[col_occ1].fillna("").astype(str)

    # --- 1. Special SIC Code Flags for col_occ1 ---
    df_out["Not_Codeable"] = s_occ1 == SPECIAL_SIC_NOT_CODEABLE
    df_out["Multiple_Possible_SICs"] = s_occ1 == SPECIAL_SIC_MULTIPLE_POSSIBLE

    # --- 2. Number of Answers ---
    df_out["num_answers"] = _calculate_num_answers(df_out, col_occ1, col_occ2, col_occ3)

    # --- 3. Digit/Character Match Flags for col_occ1 ---
    # These flags should ideally identify standard SIC patterns,
    # not the special codes like '-9' or '+4'.
    # We create a temporary series for matching that excludes special codes.
    s_occ1_for_matching = s_occ1.copy()
    s_occ1_for_matching[
        df_out["Not_Codeable"] | df_out["Multiple_Possible_SICs"]
    ] = ""  # Replace special codes with empty string for pattern matching

    match_flags_dict = _create_sic_match_flags(s_occ1_for_matching)
    for flag_name, flag_series in match_flags_dict.items():
        df_out[flag_name] = flag_series

    # --- 4. Unambiguous Flag ---
    # REVIEW: Confirm this definition of "Unambiguous"
    # Currently: 1 answer provided AND the primary SIC is a 5-digit standard code.
    if "Match_5_digits" in df_out.columns and "num_answers" in df_out.columns:
        df_out["Unambiguous"] = (df_out["num_answers"] == 1) & df_out[
            "Match_5_digits"
        ].fillna(False)
    else:
        df_out["Unambiguous"] = False
        missing_prereqs = []
        if "Match_5_digits" not in df_out.columns:
            missing_prereqs.append("Match_5_digits")
        if "num_answers" not in df_out.columns:
            missing_prereqs.append("num_answers")
        if missing_prereqs: # Only log if there are actually missing prerequisites
            logger.warning(
                "Prerequisite columns %s not found for 'Unambiguous' flag calculation. Defaulting to False.",
                missing_prereqs,
            )


    # --- 5. Convert to Pandas Nullable Boolean Type ---
    # "num_answers" is integer, others are boolean
    flag_cols_to_convert_to_boolean = [
        "Not_Codeable",
        "Multiple_Possible_SICs",
        "Match_5_digits",
        # "Match_4_digits", # Removed
        "Match_3_digits",
        "Match_2_digits",
        # "Match_1_digits", # Removed
        "Unambiguous",
    ]

    for flag_col_name in flag_cols_to_convert_to_boolean:
        if flag_col_name in df_out.columns:
            try:
                df_out[flag_col_name] = df_out[flag_col_name].astype("boolean")
            except (TypeError, ValueError) as e:
                logger.warning(
                    "Could not convert column '%s' to boolean dtype: %s",
                    flag_col_name,
                    e,
                )

    logger.info("Finished adding data quality flag columns.")
    final_flag_columns = ["num_answers"] + flag_cols_to_convert_to_boolean
    logger.debug(
        "Flag columns added/updated: %s",
        [col for col in final_flag_columns if col in df_out.columns],
    )

    if logger.isEnabledFor(logging.DEBUG):
        print(f"--- DataFrame info after adding flags (for {len(df_out)} rows) ---")
        df_out.info()
        # Example of how to see value counts for new columns
        for col_debug in final_flag_columns:
            if col_debug in df_out.columns:
                print(f"\n--- Value Counts for {col_debug} ---")
                print(df_out[col_debug].value_counts(dropna=False))
        print("----------------------------------------------------------")

    return df_out


# pylint: enable=too-many-locals, too-many-statements


def read_sic_data(
    file_path: str | Path, loaded_config: Optional[dict[str, Any]] = None
) -> pd.DataFrame:
    """Reads a comma-separated CSV file and returns a DataFrame.

    Uses column names defined in the config if provided, otherwise defaults.

    Args:
        file_path (str | Path): The path to the CSV file.
        loaded_config (Optional[dict]): Loaded configuration dictionary.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    file_path = Path(file_path)
    logger.info("Reading SIC data from: %s", file_path)

    # Define default column names if not provided or if config structure is different
    default_column_names = [
        "unique_id", "sic_section", "sic2007_employee", "sic2007_self_employed",
        "sic_ind1", "sic_ind2", "sic_ind3", "sic_ind_code_flag",
        "soc2020_job_title", "soc2020_job_description",
        "sic_ind_occ1", "sic_ind_occ2", "sic_ind_occ3", "sic_ind_occ_flag",
    ]
    # More robust way to get column names from config
    column_names_mapping = loaded_config.get("column_names", {}) if loaded_config else {}
    input_columns = column_names_mapping.get("input_columns", default_column_names)


    try:
        sic_data = pd.read_csv(
            file_path,
            delimiter=",",
            names=input_columns if not column_names_mapping.get("header_in_file", False) else None, # Use names if no header
            header=0 if column_names_mapping.get("header_in_file", False) else None, # Use header if specified
            dtype=str, # Read all as string initially
            na_filter=False, # Keep empty strings as is, don't convert to NaN
            encoding="utf-8",
        )
        logger.info("Successfully loaded %d rows from %s", len(sic_data), file_path)
        # If names were applied and there was a header, first row might be header
        if not column_names_mapping.get("header_in_file", False) and not sic_data.empty:
            # Basic check if first row looks like header when names are applied
            if sic_data.iloc[0][0] == default_column_names[0] or sic_data.iloc[0][0] == input_columns[0]: # Check first col
                 logger.info("Detected header row in data when 'names' were applied. Skipping first row.")
                 sic_data = sic_data.iloc[1:].reset_index(drop=True)

        return sic_data
    except FileNotFoundError:
        logger.exception("Data file not found: %s", file_path)
        raise
    except pd.errors.EmptyDataError:
        logger.exception("Data file is empty: %s", file_path)
        raise
    except pd.errors.ParserError:
        logger.exception("Error parsing data file: %s", file_path)
        raise
    except Exception as e: # pylint: disable=broad-except
        logger.exception("Error reading data file %s: %s", file_path, e)
        raise


# --- Example Usage (Optional: Can be in a separate main script) ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, # Set to DEBUG to see detailed df.info() and value_counts
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # Dummy config_loader for standalone execution of this script
    def load_config_dummy(config_path: str) -> Optional[dict[str, Any]]:
        logger.info("Using dummy config loader for example.")
        # Simulate loading a config. Adjust gold_standard_csv path as needed.
        # Also, define column_names if your CSV doesn't use the defaults
        # or if it has a header.
        return {
            "paths": {"gold_standard_csv": "your_data.csv"}, # <--- CHANGE THIS TO YOUR CSV FILE
            "column_names": {
                # Example: If your CSV file already has a header row
                 "header_in_file": True,
                # "input_columns": ["ID", "SIC1", "SIC2", "SIC3", ...], # if names differ
                # Default column names (from default_column_names list) will be used if
                # "input_columns" is not specified and "header_in_file" is False or not specified.
                "sic_ind_occ1": "sic_ind_occ1", # Ensure these match your actual column names
                "sic_ind_occ2": "sic_ind_occ2", # or defaults if using them.
                "sic_ind_occ3": "sic_ind_occ3",
            },
        }

    CONFIG_PATH = "config.toml" # Path for the real config loader
    main_config: Optional[dict[str, Any]] = None
    try:
        # Replace with your actual config loader if not running standalone
        main_config = load_config_dummy(CONFIG_PATH) # Using dummy for example
        # from src.config_loader import load_config # Uncomment for your project
        # main_config = load_config(CONFIG_PATH)
    except FileNotFoundError: # From actual load_config
        logger.error("Main config: File not found at %s", CONFIG_PATH)
    except ImportError:
        logger.error("Could not import 'src.config_loader'. Using dummy config for example.")
        if main_config is None: # Ensure main_config is assigned if import fails
             main_config = load_config_dummy(CONFIG_PATH)
    except Exception as e: # pylint: disable=broad-except
        logger.error("Error loading configuration: %s", e)


    if (
        main_config is not None
        and "paths" in main_config
        and "gold_standard_csv" in main_config["paths"]
    ):
        data_file_path_str = main_config["paths"]["gold_standard_csv"]
        data_file_path = Path(data_file_path_str)

        # Create a dummy CSV for testing if your_data.csv doesn't exist
        if not data_file_path.exists():
            logger.warning(
                "Test data file '%s' not found. Creating a dummy CSV for demonstration.",
                data_file_path
            )
            dummy_data = {
                "unique_id": ["ID001", "ID002", "ID003", "ID004", "ID005", "ID006", "ID007", "ID008", "ID009", "ID010"],
                "sic_ind_occ1": ["12345", "-9", "+4", "5432x", "987xx", "12xxx", "12345", "12345", "", "NA"],
                "sic_ind_occ2": ["", "NA", "11111", "", "NA", "22222", "33333", "", "55555", "NA"],
                "sic_ind_occ3": ["NA", "", "11111", "NA", "NA", "22222", "", "44444", "", "NA"],
                # Add other default columns if your read_sic_data expects them and header_in_file=False
                "sic_section": ["A"] * 10, "sic2007_employee": ["Emp"] * 10,
                "sic2007_self_employed": ["SE"] * 10, "sic_ind1": ["I1"] * 10,
                "sic_ind2": ["I2"] * 10, "sic_ind3": ["I3"] * 10,
                "sic_ind_code_flag": ["F"] * 10, "soc2020_job_title": ["JT"]*10,
                "soc2020_job_description": ["JD"]*10, "sic_ind_occ_flag": ["OF"]*10,
            }
            dummy_df = pd.DataFrame(dummy_data)
            try:
                dummy_df.to_csv(data_file_path, index=False) # With header
                logger.info("Dummy CSV created at %s", data_file_path)
                # If using dummy, ensure config reflects header_in_file=True
                if "column_names" not in main_config: main_config["column_names"] = {}
                main_config["column_names"]["header_in_file"] = True
            except IOError as e_io:
                logger.error("Could not write dummy CSV: %s", e_io)


        try:
            sic_dataframe = read_sic_data(data_file_path, main_config)

            if not sic_dataframe.empty:
                sic_dataframe_with_flags = add_data_quality_flags(
                    sic_dataframe, main_config
                )

                print("\nDataFrame Head with Quality Flags:")
                print(sic_dataframe_with_flags.head(10))

                print("\nValue Counts for Selected Quality Flags:")
                flag_cols_to_show = [
                    "num_answers",
                    "Not_Codeable",
                    "Multiple_Possible_SICs",
                    "Match_5_digits",
                    "Match_3_digits",
                    "Match_2_digits",
                    "Unambiguous",
                ]
                for col_to_show in flag_cols_to_show:
                    if col_to_show in sic_dataframe_with_flags.columns:
                        print(f"\n--- {col_to_show} ---")
                        print(
                            sic_dataframe_with_flags[col_to_show].value_counts(
                                dropna=False
                            )
                        )
            else:
                logger.warning(
                    "Loaded DataFrame is empty, skipping quality flag generation."
                )
        except FileNotFoundError:
            logger.error(
                "Gold standard data file not found at calculated path: %s",
                data_file_path,
            )
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as pe:
            logger.error(
                "Error with gold standard data file %s: %s", data_file_path, pe
            )
        except Exception as e: # pylint: disable=broad-except
             logger.error("An unexpected error occurred in the main execution block: %s", e)


    elif main_config is None:
        logger.error("Could not proceed because configuration failed to load.")
    else:
        logger.error(
            "Could not proceed without valid configuration including "
            "['paths']['gold_standard_csv']."
        )
