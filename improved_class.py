from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np


class ClassificationEvaluator:
    """
    A flexible toolkit to evaluate classification results in a DataFrame.

    This class can perform both one-to-one column comparisons and many-to-many
    list comparisons, with optional n-digit partial matching for both.
    """

    def __init__(self, df: pd.DataFrame, id_col: str):
        """
        Initializes the evaluator with the DataFrame to be analyzed.

        Args:
            df (pd.DataFrame): The DataFrame containing all data for evaluation.
            id_col (str): The name of the unique identifier column for debugging output.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("A non-empty pandas DataFrame must be provided.")
        if id_col not in df.columns:
            raise ValueError(f"The id_col '{id_col}' was not found in the DataFrame.")
        
        # Ensure data is treated as strings for reliable comparison
        self.df = df.astype(str)
        self.id_col = id_col

    def compare_columns(self, col1: str, col2: str, n: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculates the match rate between two specific columns (one-to-one).
        (This method is unchanged)
        """
        if col1 not in self.df.columns or col2 not in self.df.columns:
            raise ValueError(f"One or both columns ('{col1}', '{col2}') not found.")

        total_rows = len(self.df)
        if total_rows == 0:
            return {"match_percentage": 0.0, "match_count": 0, "total_rows": 0, "matching_ids": []}

        series1 = self.df[col1] if n is None else self.df[col1].str[:n]
        series2 = self.df[col2] if n is None else self.df[col2].str[:n]

        matches_mask = (series1 == series2) & (series1.notna()) & (series2.notna())
        
        match_count = matches_mask.sum()
        match_percentage = round(100 * match_count / total_rows, 2)
        matching_ids = self.df[matches_mask][self.id_col].tolist()

        return {
            "match_percentage": match_percentage,
            "match_count": match_count,
            "total_rows": total_rows,
            "matching_ids": matching_ids,
        }

    def evaluate_list_match(
        self, source_cols: List[str], target_cols: List[str], n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Checks if any value from source columns exists in the target columns for each row.
        Can optionally perform the check on the first n-digits of the codes.

        Args:
            source_cols (List[str]): Columns representing the predictions to check.
            target_cols (List[str]): Columns representing the ground truth labels.
            n (Optional[int]): The number of leading digits to compare.
                If None, performs a full string comparison. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary with statistics, including match
                            percentage, counts, and a list of matching IDs.
        """
        missing_cols = [col for col in source_cols + target_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        total_rows = len(self.df)
        if total_rows == 0:
            return {"match_percentage": 0.0, "match_count": 0, "total_rows": 0, "matching_ids": []}

        # The helper function now takes 'n' to perform the substring logic.
        def check_row_match(row):
            source_series = pd.Series(row[source_cols]).dropna()
            target_series = pd.Series(row[target_cols]).dropna()
            
            # If n is provided, truncate all values to the first n characters
            if n is not None:
                source_values = set(source_series.astype(str).str[:n])
                target_values = set(target_series.astype(str).str[:n])
            else:
                source_values = set(source_series)
                target_values = set(target_series)
            
            if not source_values or not target_values:
                return False
            
            return not source_values.isdisjoint(target_values)

        matches_mask = self.df.apply(check_row_match, axis=1)
        
        match_count = matches_mask.sum()
        match_percentage = round(100 * match_count / total_rows, 2)
        matching_ids = self.df[matches_mask][self.id_col].tolist()
        
        return {
            "match_percentage": match_percentage,
            "match_count": match_count,
            "total_rows": total_rows,
            "matching_ids": matching_ids,
        }
