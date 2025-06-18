import pandas as pd
import numpy as np
from typing import List

class LabelAccuracy:
    """Analyse classification accuracy for scenarios where model predictions can match any of multiple ground truth labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        id_col: str = "id",
        desc_col: str = "description",
        model_label_cols: list[str] = ["model_label_1", "model_label_2"],
        model_score_cols: list[str] = ["model_score_1", "model_score_2"],
        clerical_label_cols: list[str] = ["clerical_label_1", "clerical_label_2"],
    ):
        """Initializes with a DataFrame, immediately creating derived columns for analysis."""
        self.id_col = id_col
        self.desc_col = desc_col
        self.model_label_cols = model_label_cols
        self.model_score_cols = model_score_cols
        self.clerical_label_cols = clerical_label_cols

        # Basic validation
        required_cols = ([id_col, desc_col] + model_label_cols + model_score_cols + clerical_label_cols)
        if missing_cols := [col for col in required_cols if col not in df.columns]:
            raise ValueError(f"Missing required columns: {missing_cols}")
        if len(model_label_cols) != len(model_score_cols):
            raise ValueError("Number of model label columns must match number of score columns")

        self.df = df.copy().astype(str, errors='ignore')
        # This one method now efficiently calculates all match types
        self._add_derived_columns_vectorized()

    def _add_derived_columns_vectorized(self):
        """
        Efficiently adds computed columns for full and partial matches using a vectorized approach.
        This is much faster than using df.apply() on large datasets.
        """
        # --- Step 1: Reshape the data from wide to long format ---
        # Reshape the model predictions
        model_melted = self.df.melt(
            id_vars=[self.id_col],
            value_vars=self.model_label_cols,
            value_name="model_label"
        ).dropna(subset=["model_label"])

        # Reshape the clerical (ground truth) labels
        clerical_melted = self.df.melt(
            id_vars=[self.id_col],
            value_vars=self.clerical_label_cols,
            value_name="clerical_label"
        ).dropna(subset=["clerical_label"])

        # --- Step 2: Find IDs with at least one FULL match ---
        # Merge the two long dataframes where the ID and the label match exactly
        full_matches = pd.merge(
            model_melted,
            clerical_melted,
            left_on=[self.id_col, "model_label"],
            right_on=[self.id_col, "clerical_label"]
        )
        # Get the unique list of IDs that had a match
        full_match_ids = full_matches[self.id_col].unique()

        # --- Step 3: Find IDs with at least one 2-DIGIT match ---
        # Create the 2-digit substring columns before merging
        model_melted["model_label_2_digit"] = model_melted["model_label"].str[:2]
        clerical_melted["clerical_label_2_digit"] = clerical_melted["clerical_label"].str[:2]

        # Merge where the ID and the 2-digit substring match
        partial_matches = pd.merge(
            model_melted,
            clerical_melted,
            left_on=[self.id_col, "model_label_2_digit"],
            right_on=[self.id_col, "clerical_label_2_digit"]
        )
        partial_match_ids = partial_matches[self.id_col].unique()

        # --- Step 4: Map the results back to the original DataFrame ---
        # Create the new boolean columns
        self.df["is_correct"] = self.df[self.id_col].isin(full_match_ids)
        self.df["is_correct_2_digit"] = self.df[self.id_col].isin(partial_match_ids)

        # Also add the max_score column as before
        # Ensure score columns are numeric before finding the max
        for col in self.model_score_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df["max_score"] = self.df[self.model_score_cols].max(axis=1)

    def get_accuracy(self, threshold: float = 0.0, match_type: str = 'full') -> float:
        """
        Calculate accuracy for predictions above a confidence threshold.

        Args:
            threshold (float): Minimum confidence score threshold.
            match_type (str): The type of accuracy to calculate.
                              Options: 'full' (default) or '2-digit'.
        Returns:
            float: Accuracy as a percentage.
        """
        if match_type == '2-digit':
            correct_col = 'is_correct_2_digit'
        elif match_type == 'full':
            correct_col = 'is_correct'
        else:
            raise ValueError("match_type must be 'full' or '2-digit'")

        if correct_col not in self.df.columns:
            raise RuntimeError(f"Derived column '{correct_col}' not found. Ensure _add_derived_columns ran correctly.")

        filtered_df = self.df[self.df["max_score"] >= threshold]
        if len(filtered_df) == 0:
            return 0.0
            
        return 100 * filtered_df[correct_col].mean()
