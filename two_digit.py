def _extract_sic_division(
    sic_occ1_series: pd.Series,
    not_codeable_flag: pd.Series,
    multiple_possible_flag: pd.Series,
) -> pd.Series:
    """Extracts the first two digits (division) from the sic_ind_occ1 series.

    Args:
        sic_occ1_series (pd.Series): The Series containing sic_ind_occ1 codes (as strings).
        not_codeable_flag (pd.Series): Boolean Series indicating where sic_ind_occ1 is '-9'.
        multiple_possible_flag (pd.Series): Boolean Series indicating where sic_ind_occ1 is '+4'.

    Returns:
        pd.Series: A Series containing the first two digits as strings, or an
                   empty string if not applicable or for special codes.
    """
    logger.debug("Extracting SIC division (first two digits) from sic_ind_occ1.")
    # Default to empty string
    sic_division = pd.Series("", index=sic_occ1_series.index, dtype=str)

    # Condition for valid extraction:
    # Must start with at least two digits AND not be a special code.
    # Using .str.match() ensures we are dealing with strings.
    starts_with_two_digits = sic_occ1_series.str.match(r"^[0-9]{2}") # Matches if starts with 2+ digits
    
    # Rows eligible for extraction
    eligible_for_extraction = starts_with_two_digits & ~not_codeable_flag & ~multiple_possible_flag

    # Extract first two digits for eligible rows
    sic_division[eligible_for_extraction] = sic_occ1_series[eligible_for_extraction].str[:2]
    
    logger.debug("Finished extracting SIC division.")
    return sic_division
