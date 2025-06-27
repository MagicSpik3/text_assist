"""Module contains functionality to evaluate alignment between Clerical Coders (CC)
and Survey Assist (SA) results.

The classes are:
ColumnConfig
    A data structure to hold the name configurations for the analysis.

LabelAccuracy
    Analyse classification accuracy for scenarios where model predictions can match any of
    multiple ground truth labels.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml


@dataclass
class ColumnConfig:
    """A data structure to hold the name configurations for the analysis."""

    model_label_cols: List[str]
    model_score_cols: List[str]
    clerical_label_cols: List[str]
    id_col: str = "id"
    filter_unambiguous: bool = False


class LabelAccuracy:
    """Analyse classification accuracy for scenarios where model predictions can match any of
    multiple ground truth labels.
    """

    # Define missing value formats once as a class attribute for consistency
    _MISSING_VALUE_FORMATS = ["", " ", "nan", "None", "Null", "<NA>"]

    def __init__(self, df: pd.DataFrame, column_config: ColumnConfig):
        """Initialises with a dataframe and a configuration object."""
        self.config = column_config
        self.id_col = self.config.id_col
        self.model_label_cols = self.config.model_label_cols
        self.model_score_cols = self.config.model_score_cols
        self.clerical_label_cols = self.config.clerical_label_cols

        # --- Validation ---
        self._validate_inputs(df)

        # --- Data Preparation ---
        working_df = df.copy()

        # Handle unambiguous filter if required
        if self.config.filter_unambiguous:
            if working_df["Unambiguous"].dtype != bool:
                working_df["Unambiguous"] = (
                    working_df["Unambiguous"].str.lower().map({"true": True, "false": False})
                )
            working_df = working_df[working_df["Unambiguous"]]

        self.df = self._clean_dataframe(working_df)
        self._add_derived_columns()

    def _validate_inputs(self, df: pd.DataFrame):
        """Centralized method for all input validations."""
        required_cols = (
            [self.id_col]
            + self.model_label_cols
            + self.model_score_cols
            + self.clerical_label_cols
        )
        if self.config.filter_unambiguous:
            required_cols.append("Unambiguous")

        if missing_cols := [col for col in required_cols if col not in df.columns]:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if len(self.model_label_cols) != len(self.model_score_cols):
            raise ValueError(
                "Number of model label columns must match number of score columns"
            )

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by handling data types and missing values robustly.
        """
        # Convert all label columns to string type first
        label_cols = self.model_label_cols + self.clerical_label_cols
        df[label_cols] = df[label_cols].astype(str)
        
        # Replace all "impostor NaNs" in label columns with a true NaN
        df[label_cols] = df[label_cols].replace(self._MISSING_VALUE_FORMATS, np.nan)
        
        return df

    def _melt_and_clean(self, value_vars: List[str], value_name: str) -> pd.DataFrame:
        """Helper to reshape data from wide to long and drop any remaining NaNs."""
        melted_df = self.df.melt(
            id_vars=[self.id_col], value_vars=value_vars, value_name=value_name
        )
        # Now we only need to drop true NaNs, as cleaning was done in __init__
        return melted_df.dropna(subset=[value_name])

    def _add_derived_columns(self):
        """Adds computed columns for full and partial matches (vectorized)."""
        model_melted = self._melt_and_clean(
            self.model_label_cols, "model_label"
        )
        clerical_melted = self._melt_and_clean(
            self.clerical_label_cols, "clerical_label"
        )

        full_matches = pd.merge(
            model_melted,
            clerical_melted,
            left_on=[self.id_col, "model_label"],
            right_on=[self.id_col, "clerical_label"],
        )
        full_match_ids = full_matches[self.id_col].unique()

        model_melted["model_label_2_digit"] = model_melted["model_label"].str[:2]
        clerical_melted["clerical_label_2_digit"] = clerical_melted["clerical_label"].str[:2]

        partial_matches = pd.merge(
            model_melted,
            clerical_melted,
            left_on=[self.id_col, "model_label_2_digit"],
            right_on=[self.id_col, "clerical_label_2_digit"],
        )
        partial_match_ids = partial_matches[self.id_col].unique()

        self.df["is_correct"] = self.df[self.id_col].isin(full_match_ids)
        self.df["is_correct_2_digit"] = self.df[self.id_col].isin(partial_match_ids)

        for col in self.model_score_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df["max_score"] = self.df[self.model_score_cols].max(axis=1)

    def get_jaccard_similarity(self) -> float:
        """Calculates the average Jaccard Similarity Index across all rows."""
        
        def calculate_jaccard_for_row(row):
            # No need to clean here, as self.df is already cleaned
            model_set = set(row[self.model_label_cols].dropna())
            clerical_set = set(row[self.clerical_label_cols].dropna())
            
            if not model_set and not clerical_set:
                return 1.0

            intersection_size = len(model_set.intersection(clerical_set))
            union_size = len(model_set.union(clerical_set))

            return intersection_size / union_size if union_size > 0 else 0.0

        jaccard_scores = self.df.apply(calculate_jaccard_for_row, axis=1)
        return round(jaccard_scores.mean() * 100, 2)

    def get_candidate_contribution(self, candidate_col: str) -> Dict[str, Any]:
        """Assesses the value add of a single candidate column using vectorized operations."""
        primary_clerical_col = self.clerical_label_cols[0]
        if candidate_col not in self.df.columns or primary_clerical_col not in self.df.columns:
            raise ValueError("Candidate or primary clerical column not found.")
            
        # Create a working copy with only necessary, non-null candidate predictions
        working_df = self.df[[self.id_col, candidate_col] + self.clerical_label_cols].dropna(
            subset=[candidate_col]
        )
        total_considered = len(working_df)

        if total_considered == 0:
            return {"candidate_column": candidate_col, "total_predictions_made": 0}

        # --- Primary Match (already vectorized and fast) ---
        primary_match_mask = working_df[candidate_col] == working_df[primary_clerical_col]
        primary_match_count = primary_match_mask.sum()

        # --- Any Clerical Match (new vectorized approach) ---
        clerical_melted = working_df.melt(
            id_vars=[self.id_col, candidate_col],
            value_vars=self.clerical_label_cols,
            value_name="clerical_label"
        ).dropna(subset=['clerical_label'])

        any_match_mask = clerical_melted[candidate_col] == clerical_melted['clerical_label']
        any_match_ids = clerical_melted[any_match_mask][self.id_col].unique()
        any_match_count = len(any_match_ids)

        return {
            "candidate_column": candidate_col,
            "total_predictions_made": total_considered,
            "primary_match_percent": round(100 * primary_match_count / total_considered, 2),
            "primary_match_count": int(primary_match_count),
            "any_clerical_match_percent": round(100 * any_match_count / total_considered, 2),
            "any_clerical_match_count": int(any_match_count),
        }

    # ... all your other methods (get_accuracy, plot_confusion_heatmap, etc.) remain here ...

```

### Summary of Refactoring and Fixes

1.  **Centralized Cleaning (`__init__`)**:
    * The `__init__` method is now responsible for all upfront data preparation.
    * It calls a new private method, `_clean_dataframe`, which robustly replaces all "impostor NaNs" (`''`, `'None'`, etc.) with the true `np.nan` across all specified label columns.
    * This ensures that any method operating on `self.df` is working with clean, reliable data from the start.

2.  **Robust Jaccard Similarity (`get_jaccard_similarity`)**:
    * This method is now much simpler and more robust.
    * Because `self.df` has already been cleaned, the `.dropna()` call inside its helper function now correctly removes all forms of missing values, fixing the original bug.

3.  **Efficient Candidate Contribution (`get_candidate_contribution`)**:
    * This method has been significantly refactored for both **correctness and performance**.
    * It no longer uses the slow `.apply()` method for the "any match" calculation.
    * Instead, it now uses the highly efficient **`melt` and direct comparison** pattern. This is much faster on large datasets and is consistent with the approach in `_add_derived_columns`.
    * The initial `.dropna()` on the `candidate_col` now works correctly because the data has been pre-cleaned.

By centralizing the cleaning logic in the constructor, we've fixed the underlying "imposter NaN" problem for all methods simultaneously and made the entire class more efficient and maintainab
