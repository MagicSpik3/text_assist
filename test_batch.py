"""Unit tests for the batch processing script."""

import unittest
import json
from unittest.mock import patch, mock_open, MagicMock # Added MagicMock
import tempfile # For creating temporary files/directories
from pathlib import Path

import pandas as pd
import toml
import requests # Import requests for exception types

# Assuming your script is in 'scripts' directory and tests in 'tests'
# Adjust import path if your structure is different or how tests are run
# This path assumes tests are run from the project root.
from scripts.batch import (
    load_config,
    read_sic_data,
    process_row,
    process_test_set
)

# Sample data for tests
SAMPLE_TOML_CONFIG = """
[paths]
gold_standard_csv = "dummy/path/gold.csv"
output_filepath = "dummy/path/output.jsonl"

[parameters]
test_num = 2
test_mode = true
"""

SAMPLE_INVALID_TOML_CONFIG = """
[paths
gold_standard_csv = "bad"
"""

SAMPLE_CSV_DATA = """unique_id,sic_section,sic2007_employee,sic2007_self_employed,sic_ind1,sic_ind2,sic_ind3,sic_ind_code_flag,soc2020_job_title,soc2020_job_description,sic_ind_occ1,sic_ind_occ2,sic_ind_occ3,sic_ind_occ_flag
id1,A,EmpDesc1,SelfEmpDesc1,Ind1_1,Ind1_2,Ind1_3,Flag1,JobTitle1,JobDesc1,Occ1_1,Occ1_2,Occ1_3,OccFlag1
id2,B,EmpDesc2,SelfEmpDesc2,Ind2_1,Ind2_2,Ind2_3,Flag2,JobTitle2,JobDesc2,Occ2_1,Occ2_2,Occ2_3,OccFlag2
id3,C,EmpDesc3,SelfEmpDesc3,Ind3_1,Ind3_2,Ind3_3,Flag3,JobTitle3,JobDesc3,Occ3_1,Occ3_2,Occ3_3,OccFlag3
"""

class TestBatchProcessing(unittest.TestCase):
    """Tests for the batch processing script functions."""

    def test_load_config_success(self):
        """Test loading a valid TOML configuration file."""
        with patch("builtins.open", mock_open(read_data=SAMPLE_TOML_CONFIG)) as _mock_file:
            config = load_config("dummy_config.toml")
            self.assertIn("paths", config)
            self.assertEqual(config["paths"]["gold_standard_csv"], "dummy/path/gold.csv")
            self.assertEqual(config["parameters"]["test_num"], 2)

    def test_load_config_file_not_found(self):
        """Test FileNotFoundError when the config file is missing."""
        with patch("builtins.open", side_effect=FileNotFoundError) as _mock_file:
            with self.assertRaises(FileNotFoundError):
                load_config("non_existent_config.toml")

    def test_load_config_invalid_toml(self):
        """Test TomlDecodeError for malformed TOML data."""
        with patch("builtins.open", mock_open(read_data=SAMPLE_INVALID_TOML_CONFIG)) as _mock_file:
            with self.assertRaises(toml.TomlDecodeError):
                load_config("invalid_config.toml")

    @patch("pandas.read_csv")
    def test_read_sic_data_success(self, mock_pd_read_csv):
        """Test reading SIC data successfully."""
        mock_df = pd.DataFrame({"unique_id": ["id1"], "sic_ind1": ["Ind1_1"]})
        mock_pd_read_csv.return_value = mock_df

        df = read_sic_data("dummy_path.csv")
        mock_pd_read_csv.assert_called_once()
        # Check some key arguments passed to read_csv
        call_args, call_kwargs = mock_pd_read_csv.call_args
        self.assertEqual(call_kwargs.get("delimiter"), ",")
        self.assertEqual(call_kwargs.get("dtype"), str)
        self.assertTrue(isinstance(call_kwargs.get("names"), list))
        pd.testing.assert_frame_equal(df, mock_df)

    @patch("pandas.read_csv", side_effect=FileNotFoundError)
    def test_read_sic_data_file_not_found(self, _mock_pd_read_csv):
        """Test FileNotFoundError when SIC data file is missing."""
        with self.assertRaises(FileNotFoundError):
            read_sic_data("non_existent_sic_data.csv")

    @patch("requests.post")
    def test_process_row_success(self, mock_post):
        """Test processing a row with a successful API response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"classified": True, "sic_code": "12345"}
        mock_response.raise_for_status.return_value = None # Simulate no HTTP error
        mock_post.return_value = mock_response

        sample_row = pd.Series({
            "unique_id": "test_id1",
            "soc2020_job_title": "Test Job",
            "soc2020_job_description": "Test Description",
            "sic2007_employee": "Test Industry"
        })
        secret = "test_secret"

        result = process_row(sample_row, secret)

        mock_post.assert_called_once()
        self.assertTrue(result["classified"])
        self.assertEqual(result["sic_code"], "12345")
        self.assertEqual(result["unique_id"], "test_id1") # Check metadata added
        self.assertIn("request_payload", result)

    @patch("requests.post", side_effect=requests.exceptions.RequestException("API Error"))
    def test_process_row_request_exception(self, mock_post):
        """Test processing a row when API request fails."""
        sample_row = pd.Series({
            "unique_id": "test_id2",
            "soc2020_job_title": "Another Job",
            "soc2020_job_description": "Another Description",
            "sic2007_employee": "Another Industry"
        })
        secret = "test_secret"

        result = process_row(sample_row, secret)

        mock_post.assert_called_once()
        self.assertIn("error", result)
        self.assertEqual(result["unique_id"], "test_id2")

    @patch("requests.post")
    def test_process_row_http_error(self, mock_post):
        """Test processing a row when API returns an HTTP error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("HTTP Error")
        mock_post.return_value = mock_response

        sample_row = pd.Series({
            "unique_id": "test_id3",
            "soc2020_job_title": "Error Job",
            "soc2020_job_description": "Error Description",
            "sic2007_employee": "Error Industry"
        })
        secret = "test_secret"

        result = process_row(sample_row, secret)
        mock_post.assert_called_once()
        self.assertIn("error", result)
        self.assertTrue("HTTP Error" in result["error"])


    @patch("scripts.batch.pd.read_csv")
    @patch("scripts.batch.process_row")
    @patch("builtins.open", new_callable=mock_open)
    @patch("scripts.batch.time.sleep") # Mock time.sleep to speed up tests
    def test_process_test_set_test_mode(self, mock_sleep, mock_file_open, mock_process_row, mock_pd_read_csv):
        """Test process_test_set in test mode."""
        # Create a dummy DataFrame for pd.read_csv to return
        sample_input_df = pd.read_csv(io.StringIO(SAMPLE_CSV_DATA)) # Use io.StringIO
        mock_pd_read_csv.return_value = sample_input_df

        # Define what process_row should return
        mock_process_row.return_value = {"unique_id": "mock_id", "processed": True}

        test_limit = 2
        process_test_set(
            secret_code="test_secret",
            csv_filepath="dummy_input.csv",
            output_filepath="dummy_output.jsonl",
            test_mode=True,
            test_limit=test_limit
        )

        mock_pd_read_csv.assert_called_once_with("dummy_input.csv", delimiter=",", dtype=str)
        self.assertEqual(mock_process_row.call_count, test_limit)
        mock_file_open.assert_called_once_with("dummy_output.jsonl", "a", encoding="utf-8")
        # Check if write was called for each processed row
        self.assertEqual(mock_file_open().write.call_count, test_limit)
        # Check content of one of the writes
        expected_json_output = json.dumps({"unique_id": "mock_id", "processed": True}) + "\n"
        mock_file_open().write.assert_any_call(expected_json_output)
        self.assertEqual(mock_sleep.call_count, test_limit)


    @patch("scripts.batch.pd.read_csv")
    @patch("scripts.batch.process_row")
    @patch("builtins.open", new_callable=mock_open)
    @patch("scripts.batch.time.sleep")
    def test_process_test_set_full_mode(self, mock_sleep, mock_file_open, mock_process_row, mock_pd_read_csv):
        """Test process_test_set in full mode (not test_mode)."""
        sample_input_df = pd.read_csv(io.StringIO(SAMPLE_CSV_DATA)) # Use io.StringIO
        mock_pd_read_csv.return_value = sample_input_df
        mock_process_row.return_value = {"unique_id": "mock_id", "processed": True}

        num_rows_in_sample = len(sample_input_df)

        process_test_set(
            secret_code="test_secret",
            csv_filepath="dummy_input.csv",
            output_filepath="dummy_output.jsonl",
            test_mode=False # Full mode
            # test_limit is ignored when test_mode is False
        )

        self.assertEqual(mock_process_row.call_count, num_rows_in_sample)
        self.assertEqual(mock_file_open().write.call_count, num_rows_in_sample)
        self.assertEqual(mock_sleep.call_count, num_rows_in_sample)

    @patch("scripts.batch.pd.read_csv", side_effect=FileNotFoundError)
    @patch("scripts.batch.process_row") # To check it's not called
    @patch("builtins.open") # To check it's not called
    def test_process_test_set_input_file_not_found(self, mock_file_open, mock_process_row, mock_pd_read_csv):
        """Test process_test_set when input CSV is not found."""
        # No need to call with self.assertRaises as the function handles it internally (logs error)
        # We just check that subsequent operations are not performed.
        process_test_set(
            secret_code="test_secret",
            csv_filepath="non_existent.csv",
            output_filepath="dummy_output.jsonl",
            test_mode=True,
            test_limit=2
        )
        mock_pd_read_csv.assert_called_once_with("non_existent.csv", delimiter=",", dtype=str)
        mock_process_row.assert_not_called()
        mock_file_open.assert_not_called()


if __name__ == "__main__":
    # Import io for StringIO if not already at top level
    import io
    unittest.main()
