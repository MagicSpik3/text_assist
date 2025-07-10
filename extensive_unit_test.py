"""Unit tests for the LabelAccuracy class in the coder_alignment module.

This test suite verifies the functionality of the LabelAccuracy class, ensuring
that it correctly processes input data and calculates various evaluation metrics.
The tests cover:
- Correct initialization and data cleaning (handling of NaNs, special codes).
- Accurate creation of derived boolean columns ('is_correct', 'is_correct_2_digit').
- Validation of core metric calculations (accuracy, Jaccard similarity).
- Verification of the candidate contribution analysis.
- The ability of plotting functions to run without raising errors.
- Robustness against bad inputs and edge cases.
- Verification of utility methods like get_coverage, get_summary_stats, and save_output.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from survey_assist_utils.evaluation.coder_alignment import (
    ColumnConfig,
    LabelAccuracy,
)


# --- Fixtures (No Changes Needed) ---
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


# --- Existing Tests (Largely Unchanged) ---

def test_init_and_cleaning(sample_data_and_config):
    """Tests that the class initializes and cleans data correctly."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)
    assert analyzer.df.loc[1, "clerical_label_1"] == "01234"
    assert analyzer.df.loc[2, "clerical_label_1"] == "-9"
    assert pd.isna(analyzer.df.loc[3, "clerical_label_1"])


def test_add_derived_columns(sample_data_and_config):
    """Tests that the derived columns are created with correct values."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)
    assert analyzer.df["is_correct"].tolist() == [True, True, False, False, False]
    assert analyzer.df["is_correct_2_digit"].tolist() == [True, True, False, False, True]
    assert analyzer.df.loc[0, "max_score"] == 0.9


def test_get_jaccard_similarity(sample_data_and_config):
    """Tests the Jaccard similarity calculation."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)
    # Corrected expected calculation for Jaccard:
    # A: int=1, uni=3 -> 1/3
    # B: int=1, uni=3 -> 1/3
    # C: int=1, uni=3 -> 1/3
    # D: int=0, uni=2 -> 0
    # E: int=0, uni=4 -> 0
    # Mean = (1/3 + 1/3 + 1/3 + 0 + 0) / 5 = 1 / 5 = 0.2
    assert analyzer.get_jaccard_similarity() == pytest.approx(0.20, abs=0.01)


def test_get_candidate_contribution(sample_data_and_config):
    """Tests the candidate contribution method for a single candidate."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)
    result = analyzer.get_candidate_contribution("model_label_1")
    # model_label_1 matches clerical_label_1 once ('12345') and clerical_label_2 once (np.nan ignored)
    # clerical_label_1: "12345", "01234", "-9", nan, "5432x"
    # model_label_1: "12345", "01234", "99999", "54321", "54322"
    # clerical_label_2: "23456", nan, "4+", nan, "54321"
    # model_label_1 '12345' matches '12345' in clerical_label_1.
    # model_label_1 '01234' matches '01234' in clerical_label_1.
    assert result["primary_match_count"] == 2
    # model_label_1 '54321' matches '54321' in clerical_label_2
    assert result["any_clerical_match_count"] == 3
    assert result["any_clerical_match_percent"] == 60.0


# --- New Tests for Improved Coverage ---

def test_validate_inputs_raises_errors(sample_data_and_config):
    """Tests that __init__ raises ValueErrors for bad configuration."""
    df, _ = sample_data_and_config

    # Test for missing column
    bad_config_missing = ColumnConfig(
        model_label_cols=["model_label_1", "missing_col"],
        model_score_cols=["model_score_1", "model_score_2"],
        clerical_label_cols=["clerical_label_1", "clerical_label_2"],
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        LabelAccuracy(df=df, column_config=bad_config_missing)

    # Test for mismatched score/label columns
    bad_config_mismatch = ColumnConfig(
        model_label_cols=["model_label_1"],
        model_score_cols=["model_score_1", "model_score_2"],
        clerical_label_cols=["clerical_label_1", "clerical_label_2"],
    )
    with pytest.raises(ValueError, match="must match number of score columns"):
        LabelAccuracy(df=df, column_config=bad_config_mismatch)


def test_safe_zfill_logic():
    """Tests the _safe_zfill static method directly with various edge cases."""
    # Test padding
    assert LabelAccuracy._safe_zfill("123") == "00123"
    # Test special codes
    assert LabelAccuracy._safe_zfill("-9") == "-9"
    assert LabelAccuracy._safe_zfill("4+") == "4+"
    # Test non-numeric strings
    assert LabelAccuracy._safe_zfill("1234x") == "1234x"
    # Test NaNs
    assert pd.isna(LabelAccuracy._safe_zfill(np.nan))
    assert pd.isna(LabelAccuracy._safe_zfill(None))


def test_get_accuracy_thoroughly(sample_data_and_config):
    """Tests the get_accuracy method with more scenarios."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    # Test 2-digit match accuracy (3 correct out of 5 = 60%)
    result = analyzer.get_accuracy(match_type="2-digit", extended=True)
    assert result["accuracy_percent"] == 60.0

    # Test with a threshold that filters the data
    # At threshold 0.8, rows A, B, E remain.
    # is_correct: A=True, B=True, E=False. -> 2/3 = 66.7%
    result_thresh = analyzer.get_accuracy(threshold=0.8, extended=True)
    assert result_thresh["accuracy_percent"] == 66.7
    assert result_thresh["total_considered"] == 3

    # Test edge case where no data meets threshold
    assert analyzer.get_accuracy(threshold=1.0) == 0.0


def test_get_coverage(sample_data_and_config):
    """Tests the get_coverage method."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    # 3 of 5 scores are >= 0.8, so coverage is 60%
    assert analyzer.get_coverage(threshold=0.8) == 60.0
    # All scores are >= 0.1, so coverage is 100%
    assert analyzer.get_coverage(threshold=0.1) == 100.0


def test_get_summary_stats(sample_data_and_config):
    """Tests that get_summary_stats returns a dictionary with correct keys and values."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)
    stats = analyzer.get_summary_stats()

    assert isinstance(stats, dict)
    assert stats["total_samples"] == 5
    # Overall accuracy (full match) is 2/5 = 40%
    assert stats["overall_accuracy"] == 40.0


def test_plotting_functions_run_without_error(sample_data_and_config, monkeypatch):
    """Tests that plotting functions run without raising an error."""
    # Use monkeypatch to prevent plt.show() from blocking tests
    monkeypatch.setattr(plt, "show", lambda: None)

    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    # The test will fail automatically if any exception is raised.
    analyzer.plot_threshold_curves()
    analyzer.plot_confusion_heatmap(
        human_code_col="clerical_label_1", llm_code_col="model_label_1"
    )

    # Test edge case for heatmap where no data overlaps
    # This should print a message and return None, not raise an error.
    analyzer.plot_confusion_heatmap(
        human_code_col="clerical_label_1", llm_code_col="clerical_label_2"
    )


def test_save_output(sample_data_and_config, tmp_path):
    """Tests that save_output correctly creates files in a temporary directory."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    metadata = {"evaluation_type": "test_run", "model_version": "1.0"}
    eval_result = {"accuracy": 100.0}

    # Call the function, directing output to the temporary path provided by pytest
    output_folder = analyzer.save_output(
        metadata=metadata, eval_result=eval_result, save_path=str(tmp_path)
    )

    # Check that the folder and files were created
    assert os.path.isdir(output_folder)
    metadata_path = os.path.join(output_folder, "metadata.json")
    eval_path = os.path.join(output_folder, "evaluation_result.json")
    assert os.path.isfile(metadata_path)
    assert os.path.isfile(eval_path)

    # Check the content of one of the files
    with open(metadata_path, "r", encoding="utf-8") as f:
        saved_meta = json.load(f)
    assert saved_meta["model_version"] == "1.0"

    # Test that it raises an error for empty metadata
    with pytest.raises(ValueError, match="Metadata dictionary cannot be empty"):
        analyzer.save_output(metadata={}, eval_result=eval_result)
