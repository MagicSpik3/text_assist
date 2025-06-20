"""Contains a focused utility class for flattening LLM JSON responses."""

import json
import logging
from typing import List, Dict, Any
import pandas as pd
# from google.cloud import storage # Required for GCS integration

class JsonFlattener:
    """A utility class to flatten raw LLM JSON responses into a DataFrame."""

    def __init__(self, json_keys_config: Dict[str, Any], max_candidates: int):
        """
        Initializes the flattener with configuration for JSON keys.

        Args:
            json_keys_config: A dictionary defining the keys in the source JSON.
            max_candidates: The maximum number of candidates to flatten.
        """
        self.keys = json_keys_config
        self.max_candidates = max_candidates

    def flatten_records(self, json_data: List[Dict]) -> pd.DataFrame:
        """
        Flattens a list of raw JSON records into a clean DataFrame.

        Args:
            json_data: A list of dictionary records from the LLM output.

        Returns:
            A flattened pandas DataFrame.
        """
        if not isinstance(json_data, list):
            logging.warning("JSON data is not a list. Wrapping it in a list.")
            json_data = [json_data]
            
        # Define the structure for pandas.json_normalize for efficiency
        record_meta = [
            self.keys["unique_id"],
            self.keys["classified"],
            self.keys["followup"],
            [self.keys["source_sic_code"], "chosen_sic_code"],
            [self.keys["source_sic_description"], "chosen_sic_description"],
            self.keys["reasoning"],
            [self.keys["payload_path"], self.keys["payload_job_title"]],
            [self.keys["payload_path"], self.keys["payload_job_description"]],
        ]
        
        df = pd.json_normalize(
            json_data,
            record_path=self.keys["candidates_path"],
            meta=record_meta,
            record_prefix="candidate_",
            errors='ignore'
        )
        
        if df.empty:
            return pd.DataFrame()
            
        # Unstack the candidates to turn them into columns
        index_cols = [col for col in df.columns if not col.startswith('candidate_')]
        df = df.set_index(index_cols)
        df = df.groupby(level=index_cols).cumcount().add(1).to_frame('candidate_num').join(df)
        df = df[df['candidate_num'] <= self.max_candidates]
        df = df.unstack('candidate_num')
        df.columns = [f'{col}_{num}' for col, num in df.columns]
        
        return df.reset_index()

    def flatten_from_gcs(self, bucket_name: str, blob_name: str) -> pd.DataFrame:
        """Downloads a JSON file from GCS and flattens it."""
        logging.info("Downloading %s from bucket %s", blob_name, bucket_name)
        # client = storage.Client()
        # bucket = client.bucket(bucket_name)
        # blob = bucket.blob(blob_name)
        # try:
        #     json_data = json.loads(blob.download_as_string())
        #     return self.flatten_records(json_data)
        # except Exception as e:
        #     logging.error("Failed to download or parse GCS file: %s", e)
        #     return pd.DataFrame()
        # This is a placeholder for your actual GCS download logic
        print(f"--- SIMULATING: Would download gs://{bucket_name}/{blob_name} ---")
        # In a real scenario, you would use the commented out code above.
        # For now, let's pretend we downloaded it and have some sample data.
        sample_json_data = [{"unique_id": "sim_001", "sic_code": "12345", "sic_candidates": []}]
        return self.flatten_records(sample_json_data)
