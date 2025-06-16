import os
import json
import pytest
import pandas as pd
import toml

# Assume your JsonPreprocessor class is in a file named 'preprocessor.py'
from preprocessor import JsonPreprocessor

# --- Test Setup Fixture ---

@pytest.fixture
def mock_environment(tmp_path):
    """
    Sets up a temporary directory with mock config, JSON, and CSV files for testing.
    This fixture is automatically used by tests that list it as an argument.
    """
    # 1. Create mock directories
    json_dir = tmp_path / "json_runs"
    os.makedirs(json_dir)

    # 2. Create mock config.toml
    config_data = {
        "paths": {
            "local_json_dir": str(json_dir),
            "processed_csv_output": str(tmp_path / "eval_data.csv")
        },
        "parameters": {"date_since": "20250615"},
        "json_keys": {"unique_id": "unique_id"}
    }
    config_path = tmp_path / "config.toml"
    with open(config_path, "w") as f:
        toml.dump(config_data, f)

    # 3. Create mock JSON files
    # This file SHOULD be processed (date is >= date_since)
    json_data_1 = [
        {"unique_id": "id_01", "sic_code": "1111"},
        {"unique_id": "id_02", "sic_code": "2222"},
    ]
    with open(json_dir / "20250615_output.json", "w") as f:
        json.dump(json_data_1, f)

    # This file SHOULD ALSO be processed
    json_data_2 = [{"unique_id": "id_03", "sic_code": "3333"}]
    with open(json_dir / "20250616_output.json", "w") as f:
        json.dump(json_data_2, f)
    
    # This file SHOULD BE IGNORED (date is too old)
    with open(json_dir / "20240101_output.json", "w") as f:
        json.dump([{"unique_id": "id_99"}], f)

    # 4. Create mock evaluation CSV for merging
    eval_df = pd.DataFrame({
        "unique_id": ["id_01", "id_03"],
        "ground_truth": ["A", "C"]
    })
    eval_df.to_csv(tmp_path / "eval_data.csv", index=False)

    return config_path


# --- Unit Tests for Each Method ---

def test_load_config(mock_environment):
    """Tests that the config is loaded correctly."""
    preprocessor = JsonPreprocessor(config_path=mock_environment)
    assert preprocessor.config["parameters"]["date_since"] == "20250615"
    assert "local_json_dir" in preprocessor.config["paths"]

def test_get_local_filepaths(mock_environment):
    """Tests that only files on or after the 'date_since' are found."""
    preprocessor = JsonPreprocessor(config_path=mock_environment)
    filepaths = preprocessor.get_local_filepaths()
    
    assert len(filepaths) == 2
    assert any("20250615_output.json" in path for path in filepaths)
    assert any("20250616_output.json" in path for path in filepaths)
    assert not any("20240101_output.json" in path for path in filepaths)

def test_record_count(mock_environment):
    """Tests that the number of records in a single file is counted correctly."""
    preprocessor = JsonPreprocessor(config_path=mock_environment)
    # Get the path to one of our mock files
    json_path = os.path.join(preprocessor.config["paths"]["local_json_dir"], "20250615_output.json")
    
    count = preprocessor.record_count(json_path)
    assert count == 2

def test_flatten_llm_json_to_dataframe(mock_environment):
    """Tests that a single JSON file is flattened into a DataFrame correctly."""
    preprocessor = JsonPreprocessor(config_path=mock_environment)
    json_path = os.path.join(preprocessor.config["paths"]["local_json_dir"], "20250615_output.json")

    df = preprocessor.flatten_llm_json_to_dataframe(json_path, max_candidates=1)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "unique_id" in df.columns
    assert "chosen_sic_code" in df.columns
    assert df.iloc[0]["unique_id"] == "id_01"
    assert df.iloc[1]["chosen_sic_code"] == "2222"

def test_count_all_records(mock_environment):
    """Tests that the total number of records across all valid files is correct."""
    preprocessor = JsonPreprocessor(config_path=mock_environment)
    total_count = preprocessor.count_all_records()
    
    # 2 records from the first file + 1 record from the second file
    assert total_count == 3

def test_process_files(mock_environment):
    """Tests the full processing pipeline: find, flatten, combine, and deduplicate."""
    preprocessor = JsonPreprocessor(config_path=mock_environment)
    # Add a duplicate record to one file to test deduplication
    json_path = os.path.join(preprocessor.config["paths"]["local_json_dir"], "20250615_output.json")
    with open(json_path, "a") as f:
        json.dump([{"unique_id": "id_01"}], f) # This won't work, need to reload and append
    
    with open(json_path, 'r+') as f:
        data = json.load(f)
        data.append({"unique_id": "id_01", "sic_code": "9999"})
        f.seek(0)
        json.dump(data, f)
        
    combined_df = preprocessor.process_files()
    
    assert isinstance(combined_df, pd.DataFrame)
    # Should have 3 unique records: id_01, id_02, id_03
    assert len(combined_df) == 3
    # Check that the first instance of id_01 was kept
    assert combined_df[combined_df['unique_id'] == 'id_01']['chosen_sic_code'].iloc[0] == '1111'

def test_merge_eval_data(mock_environment):
    """Tests that the processed DataFrame merges correctly with evaluation data."""
    preprocessor = JsonPreprocessor(config_path=mock_environment)
    # First, create a flattened DataFrame to merge with
    flattened_df = preprocessor.process_files()
    
    # Now, merge it
    merged_df = preprocessor.merge_eval_data(flattened_df)
    
    assert isinstance(merged_df, pd.DataFrame)
    # The merge is an "inner" join, so it should only contain ids present in both files (id_01, id_03)
    assert len(merged_df) == 2
    assert "ground_truth" in merged_df.columns
    assert list(merged_df['unique_id']) == ['id_01', 'id_03']
