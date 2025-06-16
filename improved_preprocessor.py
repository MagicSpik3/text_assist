import logging
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pandas as pd

# For Python 3.11+, tomllib is in the standard library
if sys.version_info >= (3, 11):
    import tomllib
else:
    import toml as tomllib


class JsonPreprocessor:
    """
    Handles the loading, processing, and merging of raw LLM JSON files.
    """

    def __init__(self, config_path: str):
        """Initializes the preprocessor by loading the TOML configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        logging.info("JsonPreprocessor initialized.")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        # (This method is unchanged and remains robust)
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
        # This ensures logging is set up for the instance
        log_cfg = self.config.get("logging", {})
        logging.basicConfig(
            level=log_cfg.get("level", "INFO"),
            format=log_cfg.get("format", "%(asctime)s - %(levelname)s - %(message)s"),
        )

    def _get_filepaths_from_config(self) -> List[str]:
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
                        continue # Skip files with invalid date format
            logging.info("Found %d files to process.", len(later_files))
            return later_files
        except FileNotFoundError:
            logging.error("Source directory not found: %s", directory)
            return []

    def _read_and_parse_json(self, file_path: str) -> List[Dict]:
        """
        A single, reliable method to read a JSON file and return a list of records.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Ensure the data is a list of records for consistent processing
            if isinstance(json_data, dict):
                return [json_data]
            elif isinstance(json_data, list):
                return json_data
            else:
                logging.warning("JSON content in %s is not a list or object. Skipping.", file_path)
                return []
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error("Could not read or parse JSON from %s: %s", file_path, e)
            return []

    def _flatten_records(self, all_records: List[Dict]) -> pd.DataFrame:
        """
        Flattens a list of raw record dictionaries into a clean DataFrame
        using pandas.json_normalize for efficiency and simplicity.
        """
        keys = self.config["json_keys"]
        max_candidates = self.config["parameters"]["max_candidates"]
        
        # Define the structure for json_normalize
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
        
        # Use pandas' powerful json_normalize function
        df = pd.json_normalize(
            all_records,
            record_path=keys["candidates_path"],
            meta=record_meta,
            record_prefix="candidate_",
            errors='ignore'
        )
        
        if df.empty:
            return df
            
        # Unstack the candidates to turn them into columns
        index_cols = [col for col in df.columns if not col.startswith('candidate_')]
        df = df.set_index(index_cols)
        df = df.groupby(level=index_cols).cumcount().add(1).to_frame('candidate_num').join(df)
        df = df[df['candidate_num'] <= max_candidates]
        df = df.unstack('candidate_num')
        df.columns = [f'{col}_{num}' for col, num in df.columns]
        
        return df.reset_index()
    
    def process_and_merge_files(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        The main public method to orchestrate the entire pipeline:
        1. Finds relevant JSON files.
        2. Reads and combines all records.
        3. Flattens the combined records into a DataFrame.
        4. Drops duplicates.
        5. Merges with the evaluation dataset specified in the config.
        
        Returns:
            A tuple containing the final merged DataFrame and a dictionary of summary stats.
        """
        filepaths = self._get_filepaths_from_config()
        if not filepaths:
            return pd.DataFrame(), {"message": "No files found to process."}

        # Step 1: Read all records from all files into a single list
        all_records = []
        for path in filepaths:
            all_records.extend(self._read_and_parse_json(path))
        
        initial_record_count = len(all_records)
        logging.info(f"Total records read from {len(filepaths)} files: {initial_record_count}")

        # Step 2: Flatten the combined list of records
        flattened_df = self._flatten_records(all_records)
        
        # Step 3: Drop duplicates
        unique_id_col = self.config["json_keys"]["unique_id"]
        pre_drop_count = len(flattened_df)
        flattened_df.drop_duplicates(subset=[unique_id_col], inplace=True)
        post_drop_count = len(flattened_df)
        duplicates_dropped = pre_drop_count - post_drop_count
        logging.info(f"Dropped {duplicates_dropped} duplicate records.")

        # Step 4: Merge with the evaluation data
        try:
            eval_data_path = self.config["paths"]["batch_filepath"]
            eval_data = pd.read_csv(eval_data_path, dtype=str)
            eval_data.drop_duplicates(subset=unique_id_col, inplace=True)
            
            merged_df = pd.merge(eval_data, flattened_df, on=unique_id_col, how="inner")
            logging.info(f"Successfully merged data. Final shape: {merged_df.shape}")

        except FileNotFoundError:
            logging.error(f"Evaluation data file not found at {eval_data_path}. Returning unmerged data.")
            merged_df = flattened_df # Return the flattened data if merge fails
        except KeyError:
            logging.error(f"'{unique_id_col}' not found in evaluation data. Returning unmerged data.")
            merged_df = flattened_df

        stats = {
            "files_processed": len(filepaths),
            "total_records_read": initial_record_count,
            "duplicates_dropped": duplicates_dropped,
            "final_merged_rows": len(merged_df)
        }
        
        return merged_df, stats
