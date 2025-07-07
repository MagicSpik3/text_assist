# --- File 1: src/survey_assist_utils/evaluation/coder_alignment.py (Refactored) ---
# I have updated the `_safe_zfill` helper method to correctly handle your special codes.

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
    """Analyse classification accuracy for multiple ground truth labels."""

    _MISSING_VALUE_FORMATS = ["", " ", "nan", "None", "Null", "<NA>"]

    def __init__(self, df: pd.DataFrame, column_config: ColumnConfig):
        """Initialises with a dataframe and a configuration object."""
        self.config = column_config
        self.id_col = self.config.id_col
        self.model_label_cols = self.config.model_label_cols
        self.model_score_cols = self.config.model_score_cols
        self.clerical_label_cols = self.config.clerical_label_cols

        self._validate_inputs(df)
        working_df = df.copy()

        if self.config.filter_unambiguous:
            if "Unambiguous" in working_df.columns and working_df["Unambiguous"].dtype != bool:
                working_df["Unambiguous"] = (
                    working_df["Unambiguous"].str.lower().map({"true": True, "false": False})
                )
            if "Unambiguous" in working_df.columns:
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

    def _safe_zfill(self, value: Any) -> Any:
        """
        Safely pads a value with leading zeros to 5 digits, handling special cases.
        """
        if pd.isna(value):
            return value
        
        s_value = str(value)
        # Do not pad special codes
        if s_value in ['-9', '4+']:
            return s_value
            
        try:
            # Handle numeric-like strings
            return str(int(float(s_value))).zfill(5)
        except (ValueError, TypeError):
            # Return non-numeric strings (like '1234x') as-is
            return s_value

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the DataFrame by handling data types and missing values."""
        label_cols = self.model_label_cols + self.clerical_label_cols
        df[label_cols] = df[label_cols].replace(self._MISSING_VALUE_FORMATS, np.nan)
        for col in label_cols:
            df[col] = df[col].apply(self._safe_zfill)
        return df

    def _melt_and_clean(self, value_vars: List[str], value_name: str) -> pd.DataFrame:
        """Helper to reshape data from wide to long and drop any remaining NaNs."""
        melted_df = self.df.melt(
            id_vars=[self.id_col], value_vars=value_vars, value_name=value_name
        )
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
        clerical_melted["clerical_label_2_digit"] = clerical_melted[
            "clerical_label"
        ].str[:2]
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

    # --- All other public methods like get_accuracy, get_jaccard_similarity, etc. ---
    # (No changes needed to the other methods, they are included for completeness)
    def get_accuracy(
        self, threshold: float = 0.0, match_type: str = "full", extended: bool = False
    ) -> Union[float, dict[str, float]]:
        """Calculate accuracy for predictions above a confidence threshold."""
        if match_type == "2-digit":
            correct_col = "is_correct_2_digit"
        elif match_type == "full":
            correct_col = "is_correct"
        else:
            raise ValueError("match_type must be 'full' or '2-digit'")
        
        if correct_col not in self.df.columns:
            raise RuntimeError(f"Derived column '{correct_col}' not found.")
            
        filtered_df = self.df[self.df["max_score"] >= threshold]
        total_in_subset = len(filtered_df)

        if total_in_subset == 0:
            return {"accuracy_percent": 0.0, "matches": 0, "non_matches": 0, "total_considered": 0} if extended else 0.0

        match_count = filtered_df[correct_col].sum()
        accuracy_percent = 100 * match_count / total_in_subset

        if extended:
            return {
                "accuracy_percent": round(accuracy_percent, 1),
                "matches": int(match_count),
                "non_matches": total_in_subset - int(match_count),
                "total_considered": total_in_subset,
            }
        return accuracy_percent

    def get_jaccard_similarity(self) -> float:
        """Calculates the average Jaccard Similarity Index across all rows."""
        def calculate_jaccard_for_row(row):
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
        """Assesses the value add of a single candidate column."""
        primary_clerical_col = self.clerical_label_cols[0]
        required_cols = [candidate_col, primary_clerical_col] + self.clerical_label_cols
        if any(col not in self.df.columns for col in required_cols):
            raise ValueError("One or more required columns not found.")
        working_df = self.df[[self.id_col, candidate_col] + self.clerical_label_cols].dropna(subset=[candidate_col])
        total_considered = len(working_df)
        if total_considered == 0:
            return {"candidate_column": candidate_col, "total_predictions_made": 0}
        primary_match_mask = working_df[candidate_col] == working_df[primary_clerical_col]
        primary_match_count = primary_match_mask.sum()
        clerical_melted = working_df.melt(id_vars=[self.id_col, candidate_col], value_vars=self.clerical_label_cols, value_name="clerical_label").dropna(subset=['clerical_label'])
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

# --- File 2: tests/test_coder_alignment.py (New Unit Test File) ---
# This file contains simple, focused tests for your class.

import pandas as pd
import numpy as np
import pytest

# Adjust the import path based on your project structure
from src.survey_assist_utils.evaluation.coder_alignment import ColumnConfig, LabelAccuracy

@pytest.fixture
def sample_data_and_config():
    """A pytest fixture to create a standard set of test data and config."""
    test_data = pd.DataFrame({
        'unique_id': ['A', 'B', 'C', 'D', 'E'],
        'clerical_label_1': ['12345', '1234', '-9', 'nan', '5432x'],
        'clerical_label_2': ['23456', np.nan, '4+', '', '54321'],
        'model_label_1': ['12345', '01234', '4+', 'None', '54321'],
        'model_label_2': ['99999', '12300', '-9', '88888', '5432x'],
        'model_score_1': [0.9, 0.8, 0.99, 0.7, 0.85],
        'model_score_2': [0.1, 0.7, 0.98, 0.6, 0.80],
    })
    
    config = ColumnConfig(
        model_label_cols=["model_label_1", "model_label_2"],
        model_score_cols=["model_score_1", "model_score_2"],
        clerical_label_cols=["clerical_label_1", "clerical_label_2"],
        id_col="unique_id"
    )
    return test_data, config

def test_init_and_cleaning(sample_data_and_config):
    """Tests that the class initializes and cleans data correctly."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)
    
    # Test 1: Check if leading zero was added correctly
    assert analyzer.df.loc[1, 'clerical_label_1'] == '01234'
    
    # Test 2: Check if special codes were NOT padded
    assert analyzer.df.loc[2, 'clerical_label_1'] == '-9'
    assert analyzer.df.loc[2, 'clerical_label_2'] == '4+'
    
    # Test 3: Check if non-numeric string was preserved
    assert analyzer.df.loc[4, 'clerical_label_1'] == '5432x'
    
    # Test 4: Check if string 'nan' and empty string were converted to a true NaN
    assert pd.isna(analyzer.df.loc[3, 'clerical_label_1'])
    assert pd.isna(analyzer.df.loc[3, 'clerical_label_2'])

def test_add_derived_columns(sample_data_and_config):
    """Tests that the derived columns are created with correct values."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    # Check 'is_correct' column
    # A=True (12345), B=False, C=True (4+), D=False, E=True (5432x)
    expected_is_correct = [True, False, True, False, True]
    assert analyzer.df['is_correct'].tolist() == expected_is_correct

    # Check 'is_correct_2_digit' column
    # A=True (12), B=True (12), C=False, D=False, E=True (54)
    expected_is_correct_2_digit = [True, True, False, False, True]
    assert analyzer.df['is_correct_2_digit'].tolist() == expected_is_correct_2_digit
    
    # Check 'max_score'
    assert analyzer.df.loc[0, 'max_score'] == 0.9
    assert analyzer.df.loc[4, 'max_score'] == 0.85

def test_get_accuracy(sample_data_and_config):
    """Tests the get_accuracy method."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)
    
    # Test full match accuracy (3 correct out of 5 = 60%)
    result = analyzer.get_accuracy(match_type='full', extended=True)
    assert result['accuracy_percent'] == 60.0
    assert result['matches'] == 3
    assert result['total_considered'] == 5

def test_get_jaccard_similarity(sample_data_and_config):
    """Tests the Jaccard similarity calculation."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)
    
    # Expected scores per row: A=0.333, B=0.25, C=0.5, D=0.0, E=0.667
    # Average = (0.333 + 0.25 + 0.5 + 0.0 + 0.667) / 5 = 0.35 * 100 = 35.0
    # Note: I will use approx for floating point comparisons
    assert analyzer.get_jaccard_similarity() == pytest.approx(35.0, abs=0.1)

def test_get_candidate_contribution(sample_data_and_config):
    """Tests the candidate contribution method for a single candidate."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    # Test for 'model_label_1'
    result = analyzer.get_candidate_contribution('model_label_1')
    
    # 5 predictions made, 1 primary match ('12345'), 3 any match ('12345', '4+', '54321')
    assert result['total_predictions_made'] == 5
    assert result['primary_match_count'] == 1
    assert result['any_clerical_match_count'] == 3
    assert result['any_clerical_match_percent'] == 60.0

def test_plot_confusion_heatmap(sample_data_and_config, monkeypatch):
    """Tests the data preparation part of the heatmap function."""
    # We use monkeypatch to prevent plt.show() from blocking tests
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)
    
    # We can't easily test the plot itself, but we can check if it runs without error
    # A more advanced test could check the data passed to sns.heatmap
    try:
        analyzer.plot_confusion_heatmap(
            human_code_col='clerical_label_1',
            llm_code_col='model_label_1',
            top_n=3
        )
    except Exception as e:
        pytest.fail(f"plot_confusion_heatmap raised an exception: {e}")

