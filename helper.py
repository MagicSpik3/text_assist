

# --- Helper Function for SIC Code Matching ---
def _create_sic_match_flags(sic_series: pd.Series) -> Dict[str, pd.Series]:
    """Calculates various SIC code format match flags for a given Series.

    Args:
        sic_series (pd.Series): A pandas Series containing SIC codes as strings.
                                Missing values should be pre-filled (e.g., with '').

    Returns:
        Dict[str, pd.Series]: A dictionary where keys are flag names
                              (e.g., "Match_5_digits") and values are
                              boolean pandas Series.
    """
    flags = {}

    # Match 5 digits: ^[0-9]{5}$
    flags["Match_5_digits"] = sic_series.str.match(r"^[0-9]{5}$", na=False)

    # For partial matches (N digits + X 'x's)
    is_len_expected = sic_series.str.len() == EXPECTED_SIC_LENGTH
    x_count = sic_series.str.count("x", re.I)  # Count 'x' case-insensitively
    only_digits_or_x = sic_series.str.match(r"^[0-9xX]*$", na=False)
    non_x_part = sic_series.str.replace("x", "", case=False)
    # Ensure non_x_part is not empty before checking if it's all digits
    are_non_x_digits = (non_x_part != "") & non_x_part.str.match(r"^[0-9]*$", na=False)

    base_partial_check = is_len_expected & only_digits_or_x & are_non_x_digits

    flags["Match_4_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_4)
    flags["Match_3_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_3)
    flags["Match_2_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_2)
    flags["Match_1_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_1)

    return flags


# --- Data Quality Flagging ---
def add_data_quality_flags(
    df: pd.DataFrame, loaded_config: Optional[dict[str, Any]] = None
) -> pd.DataFrame:
    """Adds data quality flag columns to the DataFrame based on SIC/SOC codes.

    Args:
        df (pd.DataFrame): The input DataFrame (typically loaded by read_sic_data).
        loaded_config (Optional[dict]): Loaded configuration dictionary to get column names.

    Returns:
        pd.DataFrame: The DataFrame with added boolean quality flag columns.
                      Returns original DataFrame if essential columns are missing.
    """
    logger.info("Adding data quality flag columns...")
    df_out = df.copy()  # Work on a copy

    # Get column names from config or use defaults
    col_occ1 = (
        loaded_config.get("column_names", {}).get("sic_ind_occ1", "sic_ind_occ1")
        if loaded_config
        else "sic_ind_occ1"
    )
    col_occ2 = (
        loaded_config.get("column_names", {}).get("sic_ind_occ2", "sic_ind_occ2")
        if loaded_config
        else "sic_ind_occ2"
    )
    # col_occ3 was removed from the provided snippet, add back if needed for Single_Answer
    col_occ3_default = "sic_ind_occ3"  # Default if not in config
    col_occ3 = (
        loaded_config.get("column_names", {}).get("sic_ind_occ3", col_occ3_default)
        if loaded_config
        else col_occ3_default
    )

    # Check if essential columns exist
    # Add col_occ3 back to this list if it's used for Single_Answer
    required_input_cols = [col_occ1, col_occ2, col_occ3]
    if not all(col_name in df_out.columns for col_name in required_input_cols):
        missing_cols = set(required_input_cols) - set(df_out.columns)
        logger.error(
            "Input DataFrame missing columns for quality flags: %s. Skipping flag generation.",
            missing_cols,
        )
        return df  # Return original df

    # --- 1. Single Answer Flag ---
    # Assuming col_occ3 is also needed for Single_Answer as per original intent
    is_occ2_missing = df_out[col_occ2].isna() | (
        df_out[col_occ2].astype(str).str.upper() == "NA"
    )
    is_occ3_missing = df_out[col_occ3].isna() | (
        df_out[col_occ3].astype(str).str.upper() == "NA"
    )
    df_out["Single_Answer"] = is_occ2_missing & is_occ3_missing

    # --- 2. Digit/Character Match Flags for col_occ1 ---
    s_occ1 = df_out[col_occ1].fillna("").astype(str)
    match_flags_dict = _create_sic_match_flags(s_occ1)

    for flag_name, flag_series in match_flags_dict.items():
        df_out[flag_name] = flag_series

    # --- 3. Unambiguous Flag ---
    # Ensure "Match_5_digits" was created by the helper
    if "Match_5_digits" in df_out.columns:
        df_out["Unambiguous"] = df_out["Single_Answer"].fillna(False) & df_out[
            "Match_5_digits"
        ].fillna(False)
    else:
        # Handle case where Match_5_digits might not be created if s_occ1 was problematic
        # Though _create_sic_match_flags should always return it.
        df_out["Unambiguous"] = False
        logger.warning(
            "'Match_5_digits' column not found for 'Unambiguous' flag calculation."
        )

    # --- 4. Convert to Pandas Nullable Boolean Type ---
    flag_cols_list = [
        "Single_Answer",
        "Match_5_digits",
        "Match_4_digits",  # Added back Match_4_digits
        "Match_3_digits",
        "Match_2_digits",
        "Match_1_digits",  # Added back Match_1_digits
        "Unambiguous",
    ]

    for flag_col_name in flag_cols_list:
        if flag_col_name in df_out.columns:
            try:
                df_out[flag_col_name] = df_out[flag_col_name].astype("boolean")
            except (TypeError, ValueError) as e:  # Catch specific errors
                logger.warning(
                    "Could not convert column '%s' to boolean dtype: %s",
                    flag_col_name,
                    e,
                )

    logger.info("Finished adding data quality flag columns.")
    logger.debug("Flag columns added: %s", flag_cols_list)

    if logger.isEnabledFor(logging.DEBUG):
        print(f"--- DataFrame info after adding flags (for {len(df_out)} rows) ---")
        df_out.info()
        print("----------------------------------------------------------")

    return df_out
