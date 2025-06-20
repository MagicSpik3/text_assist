"""
This module contains the ExperimentRunner class, which orchestrates an
evaluation run based on a TOML configuration file.
"""
import toml
import json
import pandas as pd
import logging
from datetime import datetime

# Assuming the refactored flattener is in preprocessor.py
from preprocessor import JsonFlattener

class ExperimentRunner:
    """Orchestrates a single evaluation run."""

    def __init__(self, experiment_config_path: str):
        """Loads the experiment configuration."""
        logging.info("Loading experiment config from: %s", experiment_config_path)
        with open(experiment_config_path, "rb") as f:
            self.exp_config = toml.load(f)
        
        # Also load the main application config for JSON keys
        with open("config.toml", "rb") as f:
            app_config = toml.load(f)
            self.json_keys = app_config["json_keys"]
            self.max_candidates = app_config["parameters"]["max_candidates"]

    def run(self):
        """Executes the full pipeline for the experiment."""
        exp_id = self.exp_config["experiment"]["id"]
        
        # --- 1. Trigger the Docker container (Simulated) ---
        # In a real pipeline, you would use subprocess or an orchestrator
        # to run the Docker image defined in the config.
        print(f"\n--- SIMULATING: Running Docker image '{self.exp_config['process']['docker_image']}' ---")
        print(f"   - Using input: {self.exp_config['input']['dataset_uri']}")
        
        # The output JSON filename is derived from the experiment ID for traceability
        output_filename = f"{exp_id}_output.json"
        output_blob_name = f"{self.exp_config['output']['gcs_prefix']}{output_filename}"
        
        print(f"   - Writing output to: gs://{self.exp_config['output']['gcs_bucket_name']}/{output_blob_name}")
        # --- End of Simulation ---
        
        # --- 2. Process the JSON output ---
        # Initialize our focused utility class with the necessary config
        flattener = JsonFlattener(self.json_keys, self.max_candidates)
        flattened_df = flattener.flatten_from_gcs(
            bucket_name=self.exp_config['output']['gcs_bucket_name'],
            blob_name=output_blob_name
        )
        
        if flattened_df.empty:
            logging.error("Flattening JSON from GCS resulted in an empty DataFrame. Aborting.")
            return

        # --- 3. Merge with original input data ---
        # In a real scenario, you'd download the input CSV from GCS
        input_uri = self.exp_config['input']['dataset_uri']
        print(f"\n--- SIMULATING: Reading original input data from {input_uri} ---")
        # For this example, let's create a dummy input dataframe
        original_input_df = pd.DataFrame({"unique_id": ["sim_001"], "human_coded_sic": ["54321"]})
        
        merged_df = pd.merge(
            original_input_df,
            flattened_df,
            on=self.json_keys["unique_id"],
            how="inner"
        )
        
        # --- 4. Save final CSV and metadata receipt ---
        output_path = self.exp_config['output']['final_evaluation_csv']
        logging.info("Saving final merged CSV for evaluation to: %s", output_path)
        merged_df.to_csv(output_path, index=False)
        
        # Save a copy of the experiment config as a receipt for perfect traceability
        metadata_path = output_path.replace(".csv", "_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.exp_config, f, indent=4)
            
        print(f"\n✅ Experiment '{exp_id}' complete.")
        print(f"   - Final evaluation-ready data saved to: {output_path}")
        print(f"   - Metadata receipt saved to: {metadata_path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # To run a new test, you just point to its experiment file
    runner = ExperimentRunner(experiment_config_path="experiment.toml")
    runner.run()
