# Configuration for Preprocessing and Evaluation

[paths]
# Local directory where raw JSON files are stored
local_json_dir = "data/json_runs"
# GCS bucket and folder for raw JSON files
gcs_bucket_name = "your-gcs-bucket-name"
gcs_prefix = "analysis_outputs/"
# Output for the final processed CSV
processed_csv_output = "data/analysis_outputs/llm_output_master.csv"

[parameters]
# Process only files created on or after this date (YYYYMMDD)
date_since = "20250522"
# Max number of LLM candidates to process per record
max_candidates = 5

[json_keys]
# --- Top-level keys from the source JSON ---
unique_id = "unique_id"
classified = "classified"
followup = "followup"
source_sic_code = "sic_code"
source_sic_description = "sic_description"
reasoning = "reasoning"

# --- Nested payload keys ---
payload_path = "request_payload"
payload_job_title = "job_title"
payload_job_description = "job_description"

# --- Nested candidate keys ---
candidates_path = "sic_candidates"
candidate_sic_code = "sic_code"
candidate_description = "sic_descriptive"
candidate_likelihood = "likelihood"

[logging]
level = "INFO"
format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

import configparser
import json
import logging
import os
from datetime import datetime
from typing import List

import pandas as pd
# from google.cloud import storage # Uncomment if using GCS functionality


class JsonPreprocessor:
    """Handles the processing of raw LLM JSON files into a clean DataFrame."""

    def __init__(self, config_path: str):
        """Initializes the preprocessor by loading the configuration."""
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self._setup_logging()
        logging.info("JsonPreprocessor initialized with config from %s", config_path)

    def _setup_logging(self):
        """Configures logging based on the config file."""
        log_cfg = self.config["logging"]
        logging.basicConfig(
            level=log_cfg["level"],
            format=log_cfg["format"],
        )

    def _get_local_filepaths(self) -> List[str]:
        """Gets a list of local filepaths to process based on config."""
        paths_cfg = self.config["paths"]
        params_cfg = self.config["parameters"]

        directory = paths_cfg["local_json_dir"]
        date_str = params_cfg["date_since"]
        given_date = datetime.strptime(date_str, "%Y%m%d")
        
        later_files = []
        logging.info("Searching for files in '%s' on or after %s", directory, date_str)
        try:
            for filename in os.listdir(directory):
                if filename.endswith("_output.json"):
                    try:
                        file_date = datetime.strptime(filename[:8], "%Y%m%d")
                        if file_date >= given_date:
                            later_files.append(os.path.join(directory, filename))
                    except ValueError:
                        continue # Skip files with invalid date format
            logging.info("Found %d files to process.", len(later_files))
            return later_files
        except FileNotFoundError:
            logging.error("Source directory not found: %s", directory)
            return []


    def process_files(self) -> pd.DataFrame:
        """
        Main method to load, flatten, and combine all specified JSON files.

        Returns:
            pd.DataFrame: A single DataFrame containing all processed data.
        """
        filepaths = self._get_local_filepaths()
        if not filepaths:
            logging.warning("No files found to process. Returning empty DataFrame.")
            return pd.DataFrame()
        
        # More efficient: append to list and concat once at the end
        all_dfs = [self.flatten_json_to_df(path) for path in filepaths]
        
        if not all_dfs:
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        logging.info("Combined all files into a DataFrame with shape %s", combined_df.shape)
        
        # Drop duplicates based on the unique_id specified in the config
        unique_id_col = self.config["json_keys"]["unique_id"]
        combined_df.drop_duplicates(subset=[unique_id_col], inplace=True)
        logging.info("Shape after dropping duplicates: %s", combined_df.shape)
        
        return combined_df

    def flatten_json_to_df(self, file_path: str) -> pd.DataFrame:
        """
        Reads and flattens a single JSON file using pandas.json_normalize.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            pd.DataFrame: A flattened DataFrame.
        """
        keys = self.config["json_keys"]
        params = self.config["parameters"]
        max_candidates = int(params["max_candidates"])

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error("Could not read or parse %s: %s", file_path, e)
            return pd.DataFrame()

        # Handle cases where the file contains a single object instead of a list
        if isinstance(data, dict):
            data = [data]

        # Define the structure for json_normalize
        record_meta = [
            keys["unique_id"],
            keys["classified"],
            keys["followup"],
            # Rename source keys to avoid clashing with flattened candidate keys
            [keys["source_sic_code"], "chosen_sic_code"],
            [keys["source_sic_description"], "chosen_sic_description"],
            keys["reasoning"],
            [keys["payload_path"], keys["payload_job_title"]],
            [keys["payload_path"], keys["payload_job_description"]],
        ]
        
        # Use pandas' powerful json_normalize function
        df = pd.json_normalize(
            data,
            record_path=keys["candidates_path"],
            meta=record_meta,
            record_prefix="candidate_",
            errors='ignore'
        )
        
        if df.empty:
            return df
            
        # Unstack the candidates to turn them into columns
        df = df.set_index([col for col in df.columns if not col.startswith('candidate_')])
        df = df.groupby(level=0).cumcount().add(1).to_frame('candidate_num').join(df)
        df = df[df['candidate_num'] <= max_candidates]
        df = df.unstack('candidate_num')

        # Flatten the multi-level column names
        df.columns = [f'{col}_{num}' for col, num in df.columns]
        return df.reset_index()


if __name__ == "__main__":
    # This block allows the script to be run directly
    CONFIG_PATH = "config.ini"
    
    preprocessor = JsonPreprocessor(config_path=CONFIG_PATH)
    master_df = preprocessor.process_files()

    if not master_df.empty:
        output_path = preprocessor.config["paths"]["processed_csv_output"]
        master_df.to_csv(output_path, index=False)
        logging.info("✅ Successfully saved processed data to %s", output_path)
