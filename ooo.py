import pandas as pd

class SicAnalysis:
    """
    A class to handle the analysis of SIC code matching.

    This object loads, preprocesses, and calculates metrics on SIC code data,
    encapsulating all the related logic in one place.
    """
    def __init__(self, filepath: str):
        """
        Initializes the analyzer by loading and preprocessing the data.

        Args:
            filepath (str): The path to the CSV file to be analyzed.
        """
        # The data is stored as a "private" attribute within the object
        self._data = pd.read_csv(filepath, dtype=str)
        print(f"Successfully loaded data with shape: {self._data.shape}")
        
        # The constructor immediately calls the preprocessing method
        self._add_helper_columns()

    def _add_helper_columns(self):
        """A private helper method to perform the initial data setup."""
        print("Adding helper columns for analysis...")
        code_lengths = [2, 3, 4]

        # Generate substrings for 'chosen_sic_code'
        for n in code_lengths:
            self._data[f"code_{n}"] = self._data["chosen_sic_code"].str[:n]

        # Generate substrings for 'sic_ind_occ1'
        for n in code_lengths:
            self._data[f"code_cc_{n}"] = self._data["sic_ind_occ1"].str[:n]

    def calculate_first_choice_rate(self) -> float:
        """
        Calculates the percentage of exact matches between 'sic_ind_occ1'
        and 'chosen_sic_code'.
        """
        # The method now operates on self._data, so it doesn't need an argument
        data = self._data

        # The total number of records is simply the length of the DataFrame.
        # This is clearer and less error-prone than the original calculation.
        total = len(data)

        # Calculate matching values
        matching = (data["sic_ind_occ1"] == data["chosen_sic_code"]).sum()

        # Calculate percentage
        matching_percent = round(100 * matching / total, 1) if total > 0 else 0.0

        return matching_percent

    # You can easily add more metric methods here!
    def calculate_match_rate_at_n(self, n: int) -> float:
        """Calculates the match rate for the first N digits of the SIC code."""
        data = self._data
        total = len(data)

        match_col_1 = f"code_{n}"
        match_col_2 = f"code_cc_{n}"

        if match_col_1 not in data.columns or match_col_2 not in data.columns:
            raise ValueError(f"Helper columns for n={n} do not exist.")

        matching = (data[match_col_1] == data[match_col_2]).sum()
        return round(100 * matching / total, 1) if total > 0 else 0.0
