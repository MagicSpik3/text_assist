import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd

# For Python 3.11+, tomllib is in the standard library.
# This pattern provides backward compatibility.
if sys.version_info >= (3, 11):
    import tomllib
else:
    # For older Python, you need to 'pip install toml'
    import toml as tomllib


class JsonPreprocessor:
    """Handles the processing of raw LLM JSON files into a clean DataFrame."""

    def __init__(self, config_path: str):
        """Initializes the preprocessor by loading the TOML configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        logging.info("JsonPreprocessor initialized with config from %s", config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads configuration settings from a .toml file."""
        try:
            with open(config_path, "rb") as file:
                return tomllib.load(file)
        except FileNotFoundError:
            logging.error("Configuration file not found at '%s'", config_path)
            raise
        except tomllib.TOMLDecodeError as e:
            logging.error("Could not parse the TOML file: %s", e)
            raise

    def _setup_logging(self):
        """Configures logging based on the config file."""
        log_cfg = self.config["logging"]
        logging.basicConfig(level=log_cfg["level"], format=log_cfg["format"])

    def _get_local_filepaths(self) -> List[str]:
        """Gets a list of local filepaths to process based on config."""
        directory = self.config["paths"]["local_json_dir"]
        date_str = self.config["parameters"]["date_since"]
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
                        continue
            logging.info("Found %d files to process.", len(later_files))
            return later_files
        except FileNotFoundError:
            logging.error("Source directory not found: %s", directory)
            return []

    def process_files(self) -> pd.DataFrame:
        """Main method to load, flatten, and combine all specified JSON files."""
        filepaths = self._get_local_filepaths()
        if not filepaths:
            logging.warning("No files found to process. Returning empty DataFrame.")
            return pd.DataFrame()

        all_dfs = [self.flatten_json_to_df(path) for path in filepaths]
        
        # Filter out empty dataframes that may result from processing errors
        all_dfs = [df for df in all_dfs if not df.empty]

        if not all_dfs:
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        logging.info("Combined all files into a DataFrame with shape %s", combined_df.shape)
        
        unique_id_col = self.config["json_keys"]["unique_id"]
        combined_df.drop_duplicates(subset=[unique_id_col], inplace=True)
        logging.info("Shape after dropping duplicates: %s", combined_df.shape)
        
        return combined_df

    def flatten_json_to_df(self, file_path: str) -> pd.DataFrame:
        """Reads and flattens a single JSON data file using pandas.json_normalize."""
        keys = self.config["json_keys"]
        max_candidates = self.config["parameters"]["max_candidates"]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # --- THIS IS THE CORRECTED LINE ---
                # This method should ONLY ever load JSON.
                data = json.load(f)
        # --- The except block is also corrected to only handle JSON errors ---
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error("Could not read or parse JSON from %s: %s", file_path, e)
            return pd.DataFrame()

        if isinstance(data, dict):
            data = [data]

        record_meta = [
            keys["unique_id"],
            keys["classified"],
            keys["followup"],
            [keys["source_sic_code"], "chosen_sic_code"],
            [keys["source_sic_description"], "chosen_sic_description"],
            keys["reasoning"],
            [keys["payload_path"], keys["payload_job_title"]],
            [keys["payload_path"], keys["payload_job_description"]],
        ]
        
        df = pd.json_normalize(
            data,
            record_path=keys["candidates_path"],
            meta=record_meta,
            record_prefix="candidate_",
            errors='ignore'
        )
        
        if df.empty:
            return df
            
        df = df.set_index([col for col in df.columns if not col.startswith('candidate_')])
        df = df.groupby(level=0).cumcount().add(1).to_frame('candidate_num').join(df)
        df = df[df['candidate_num'] <= max_candidates]
        df = df.unstack('candidate_num')
        df.columns = [f'{col}_{num}' for col, num in df.columns]
        return df.reset_index()


if __name__ == "__main__":
    CONFIG_PATH = "config.toml"
    
    preprocessor = JsonPreprocessor(config_path=CONFIG_PATH)
    master_df = preprocessor.process_files()

    if not master_df.empty:
        output_path = preprocessor.config["paths"]["processed_csv_output"]
        master_df.to_csv(output_path, index=False)
        logging.info("✅ Successfully saved processed data to %s", output_path)
