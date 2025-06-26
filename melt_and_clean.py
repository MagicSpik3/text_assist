import pandas as pd
import numpy as np
from typing import List, Dict, Any

class LabelAccuracy:
    # Assuming the rest of your class (__init__, etc.) is here

    def _melt_and_clean(self, value_vars: List[str], value_name: str) -> pd.DataFrame:
        """
        A helper function to reshape, clean, and prepare data for matching.

        It takes a list of columns, melts them into a long format, replaces
        various missing value formats with a standard NaN, and drops them.

        Args:
            value_vars (List[str]): The columns to melt (e.g., model or clerical cols).
            value_name (str): The name to give the new value column (e.g., 'model_label').

        Returns:
            pd.DataFrame: A cleaned, long-format DataFrame with id and value columns.
        """
        missing_value_formats = ["", " ", "nan", "None", "Null", "<NA>"]
        
        # Melt the specified columns
        melted_df = self.df.melt(
            id_vars=[self.id_col],
            value_vars=value_vars,
            value_name=value_name,
        )
        
        # Replace all non-standard missing values and drop them in one chain
        melted_df[value_name] = melted_df[value_name].replace(missing_value_formats, np.nan)
        cleaned_df = melted_df.dropna(subset=[value_name])
        
        return cleaned_df

    def _add_derived_columns(self):
        """Adds computed columns for full and partial matches (vectorized)."""
        
        # --- Step 1: Reshape data using the new helper function ---
        # This is now much cleaner and avoids repetition.
        model_melted = self._melt_and_clean(
            value_vars=self.model_label_cols, value_name="model_label"
        )
        clerical_melted = self._melt_and_clean(
            value_vars=self.clerical_label_cols, value_name="clerical_label"
        )

        # --- Step 2: Find IDs with at least one FULL match ---
        # (This logic remains the same)
        full_matches = pd.merge(
            model_melted,
            clerical_melted,
            left_on=[self.id_col, "model_label"],
            right_on=[self.id_col, "clerical_label"],
        )
        full_match_ids = full_matches[self.id_col].unique()

        # --- Step 3: Find IDs with at least one 2-DIGIT match ---
        # (This logic remains the same)
        model_melted["model_label_2_digit"] = model_melted["model_label"].str[:2]
        clerical_melted["clerical_label_2_digit"] = clerical_melted["clerical_label"].str[:2]

        partial_matches = pd.merge(
            model_melted,
            clerical_melted,
            left_on=[self.id_col, "model_label_2_digit"],
            right_on=[self.id_col, "clerical_label_2_digit"],
        )
        partial_match_ids = partial_matches[self.id_col].unique()

        # --- Step 4: Map results and add max_score column ---
        # (This logic remains the same)
        self.df["is_correct"] = self.df[self.id_col].isin(full_match_ids)
        self.df["is_correct_2_digit"] = self.df[self.id_col].isin(partial_match_ids)
        
        for col in self.model_score_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df["max_score"] = self.df[self.model_score_cols].max(axis=1)
