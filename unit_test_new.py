import configparser
import json
import os
import pytest
import pandas as pd

# Make sure the JsonPreprocessor class is importable
# This might need to be adjusted based on your project structure.
# e.g., from src.preprocessing import JsonPreprocessor
from preprocessor import JsonPreprocessor


@pytest.fixture
def mock_config_and_data(tmp_path):
    """
    Pytest fixture to create a temporary config file and mock JSON data
    for a self-contained test environment.
    """
    # 1. Create mock directory structure
    config_dir = tmp_path
    json_dir = tmp_path / "test_data" / "json_runs"
    os.makedirs(json_dir)

    # 2. Create and write a mock config.ini file
    config = configparser.ConfigParser()
    config["paths"] = {
        "local_json_dir": str(json_dir),
        "processed_csv_output": str(tmp_path / "master.csv"),
    }
    config["parameters"] = {
        "date_since": "20250613",  # Will only pick up today's file
        "max_candidates": 2,
    }
    config["json_keys"] = {
        "unique_id": "unique_id",
        "classified": "classified",
        "followup": "followup",
        "source_sic_code": "sic_code",
        "source_sic_description": "sic_description",
        "reasoning": "reasoning",
        "payload_path": "request_payload",
        "payload_job_title": "job_title",
        "payload_job_description": "job_description",
        "candidates_path": "sic_candidates",
    }
    config["logging"] = {
        "level": "CRITICAL",  # Suppress logs during testing
        "format": "%(message)s",
    }
    
    config_path = config_dir / "test_config.ini"
    with open(config_path, "w") as f:
        config.write(f)

    # 3. Create mock JSON files
    # This file SHOULD be processed (date is today)
    json_data_1 = [
        {
            "unique_id": "id_001",
            "classified": True,
            "followup": False,
            "sic_code": "1111",
            "sic_description": "Top Choice",
            "reasoning": "Because.",
            "request_payload": {"job_title": "Engineer", "job_description": "Builds things"},
            "sic_candidates": [
                {"sic_code": "1111", "sic_descriptive": "Desc 1"},
                {"sic_code": "2222", "sic_descriptive": "Desc 2"},
            ],
        },
        # This record is a duplicate and should be dropped
        {
            "unique_id": "id_001",
            "classified": True,
            "followup": False,
            "sic_code": "9999", # Different code to confirm it gets dropped
            "sic_description": "Duplicate",
            "reasoning": "Duplicate.",
            "request_payload": {"job_title": "Manager", "job_description": "Manages things"},
            "sic_candidates": [
                {"sic_code": "9999", "sic_descriptive": "Desc 9"},
            ],
        },
    ]
    with open(json_dir / "20250613_120000_output.json", "w") as f:
        json.dump(json_data_1, f)
        
    # This file SHOULD be IGNORED (date is too old)
    json_data_2 = [{"unique_id": "id_002"}] # Content doesn't matter
    with open(json_dir / "20240101_120000_output.json", "w") as f:
        json.dump(json_data_2, f)
        
    return config_path


def test_preprocessor_end_to_end(mock_config_and_data):
    """
    Full integration test for the JsonPreprocessor class.
    Tests file discovery, flattening, combining, and deduplication.
    """
    # ARRANGE: The mock_config_and_data fixture has already set everything up.
    config_path = mock_config_and_data

    # ACT: Run the processor
    preprocessor = JsonPreprocessor(config_path=config_path)
    result_df = preprocessor.process_files()

    # ASSERT: Check if the output is correct
    
    # 1. Shape Assertion: Should only have 1 final record (id_001) after deduplication
    assert result_df.shape[0] == 1, "Should have dropped the duplicate unique_id"
    
    # 2. Correct File Processing Assertion
    assert "id_002" not in result_df["unique_id"].values, "Should not have processed the old file"
    
    # 3. Flattening and Renaming Assertions
    assert "chosen_sic_code" in result_df.columns
    assert result_df.iloc[0]["chosen_sic_code"] == "1111"
    
    # 4. Candidate Unstacking Assertions
    # Check that candidate 1's columns were created and have the right data
    assert "candidate_sic_code_1" in result_df.columns
    assert result_df.iloc[0]["candidate_sic_code_1"] == "1111"
    
    # Check that candidate 2's columns were created and have the right data
    assert "candidate_sic_descriptive_2" in result_df.columns
    assert result_df.iloc[0]["candidate_sic_descriptive_2"] == "Desc 2"
    
    # Check that a column for a non-existent 3rd candidate was NOT created
    assert "candidate_sic_code_3" not in result_df.columns
