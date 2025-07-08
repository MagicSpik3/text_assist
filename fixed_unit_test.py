"""Unit tests for the LabelAccuracy class in the coder_alignment module.

This test suite verifies the functionality of the LabelAccuracy class, ensuring
that it correctly processes input data and calculates various evaluation metrics.
The tests cover:
- Correct initialization and data cleaning (handling of NaNs, special codes).
- Accurate creation of derived boolean columns ('is_correct', 'is_correct_2_digit').
- Validation of core metric calculations (accuracy, Jaccard similarity).
- Verification of the candidate contribution analysis.
- The ability of plotting functions to run without raising errors.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Adjust the import path based on your project structure
from src.survey_assist_utils.evaluation.coder_alignment import (
    ColumnConfig,
    LabelAccuracy,
)


@pytest.fixture
def sample_data_and_config():
    """A pytest fixture to create a standard set of test data and config."""
    test_data = pd.DataFrame(
        {
            "unique_id": ["A", "B", "C", "D", "E"],
            "clerical_label_1": ["12345", "1234", "-9", "nan", "5432x"],
            "clerical_label_2": ["23456", np.nan, "4+", "", "54321"],
            "model_label_1": ["12345", "01234", "99999", "54321", "54322"],
            "model_label_2": ["99999", "12300", "54322", "88888", "54322"],
            "model_score_1": [0.9, 0.8, 0.99, 0.7, 0.85],
            "model_score_2": [0.1, 0.7, 0.98, 0.6, 0.80],
        }
    )

    config = ColumnConfig(
        model_label_cols=["model_label_1", "model_label_2"],
        model_score_cols=["model_score_1", "model_score_2"],
        clerical_label_cols=["clerical_label_1", "clerical_label_2"],
        id_col="unique_id",
    )
    return test_data, config


def test_init_and_cleaning(sample_data_and_config):  # pylint: disable=redefined-outer-name
    """Tests that the class initializes and cleans data correctly."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    # Test 1: Check if leading zero was added correctly
    assert analyzer.df.loc[1, "clerical_label_1"] == "01234"

    # Test 2: Check if special codes were NOT padded
    assert analyzer.df.loc[2, "clerical_label_1"] == "-9"
    assert analyzer.df.loc[2, "clerical_label_2"] == "4+"

    # Test 3: Check if non-numeric string was preserved
    assert analyzer.df.loc[4, "clerical_label_1"] == "5432x"

    # Test 4: Check if string 'nan' and empty string were converted to a true NaN
    assert pd.isna(analyzer.df.loc[3, "clerical_label_1"])
    assert pd.isna(analyzer.df.loc[3, "clerical_label_2"])


def test_add_derived_columns(sample_data_and_config):  # pylint: disable=redefined-outer-name
    """Tests that the derived columns are created with correct values."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    # Check 'is_correct' column
    # A=True (12345), B=True (01234), C=False, D=True (54321), E=False
    expected_is_correct = [True, True, False, True, False]
    assert analyzer.df["is_correct"].tolist() == expected_is_correct

    # Check 'is_correct_2_digit' column
    # A=True (12), B=True (12), C=False, D=True (54), E=True (54)
    expected_is_correct_2_digit = [True, True, False, True, True]
    assert analyzer.df["is_correct_2_digit"].tolist() == expected_is_correct_2_digit

    # Check 'max_score'
    assert analyzer.df.loc[0, "max_score"] == 0.9
    assert analyzer.df.loc[4, "max_score"] == 0.85


def test_get_accuracy(sample_data_and_config):  # pylint: disable=redefined-outer-name
    """Tests the get_accuracy method."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    # Test full match accuracy (3 correct out of 5 = 60%)
    result = analyzer.get_accuracy(match_type="full", extended=True)
    assert result["accuracy_percent"] == 60.0
    assert result["matches"] == 3
    assert result["total_considered"] == 5


def test_get_jaccard_similarity(sample_data_and_config):  # pylint: disable=redefined-outer-name
    """Tests the Jaccard similarity calculation."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    # Expected scores per row: A=0.333, B=0.333, C=0.0, D=0.25, E=0.25
    # Average = (0.333 + 0.333 + 0.0 + 0.25 + 0.25) / 5 = 0.233 * 100 = 23.3%
    assert analyzer.get_jaccard_similarity() == pytest.approx(23.33, abs=0.1)


def test_get_candidate_contribution(sample_data_and_config):  # pylint: disable=redefined-outer-name
    """Tests the candidate contribution method for a single candidate."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    # Test for 'model_label_1'
    result = analyzer.get_candidate_contribution("model_label_1")

    # 5 predictions made.
    # Primary matches: '12345' (A), '01234' (B) -> 2
    # Any matches: '12345' (A), '01234' (B), '54321' (D) -> 3
    assert result["total_predictions_made"] == 5
    assert result["primary_match_count"] == 2
    assert result["any_clerical_match_count"] == 3
    assert result["any_clerical_match_percent"] == 60.0


def test_plot_confusion_heatmap(sample_data_and_config, monkeypatch):  # pylint: disable=redefined-outer-name
    """Tests that the heatmap function runs without raising an error."""
    # We use monkeypatch to prevent plt.show() from blocking tests
    monkeypatch.setattr(plt, "show", lambda: None)

    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    # The test will fail automatically if any exception is raised.
    analyzer.plot_confusion_heatmap(
        human_code_col="clerical_label_1", llm_code_col="model_label_1", top_n=3
    )
