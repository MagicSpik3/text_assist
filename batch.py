"""This script processes SIC code batch data through Survey Assist API.
It is based on configurations specified in a .toml file.
Prior to invocation, ensure to run the following CLI commands:
> gcloud config set project "valid-priject-name"
> gcloud auth application-default login

By default the output file is appended to, not overwritten.
Delete it before running, or change the name if this is not want you want.

Run from the root of the project as follows:
poetry run python scripts/batch.py path/to/config.toml

It also requires the following environment variables to be exported:
- API_GATEWAY: The API gateway URL (Note: now configurable via toml [api_settings].base_url)
- SA_EMAIL: The service account email.
- JWT_SECRET: The path to the JWT secret.

The .toml configuration file should include:
- paths.gold_standard_csv: Path to the input data file.
- paths.output_filepath: Path for the output results.
- parameters.test_num: Number of items for test mode.
- parameters.test_mode: Boolean to enable/disable test mode.
- column_names.payload_unique_id: Column name for unique ID in input.
- column_names.payload_job_title: Column name for job title in input.
- column_names.payload_job_description: Column name for job description in input.
- column_names.payload_industry_description: Column name for industry in input.
- api_settings.base_url: The base URL for the classification API.

Usage:
    poetry run python scripts/batch.py <path_to_config.toml>

Example .toml configuration file (config.toml):
    [paths]
    gold_standard_csv = "data/all_examples_comma.csv"
    output_filepath = "data/output.jsonl"

    [parameters]
    test_num = 3
    test_mode = true

    [column_names]
    payload_unique_id = "unique_id"
    payload_job_title = "soc2020_job_title"
    # ... other column names ...

    [api_settings]
    base_url = "https://your-api-url/classify"
"""

import json
import logging
import os
import time
import argparse # For command-line arguments
from pathlib import Path

import pandas as pd
import requests
import toml # Or tomllib for Python 3.11+

# load the utils:
try:
    from utils.api_token.jwt_utils import check_and_refresh_token
except ImportError:
    logging.error("Could not import check_and_refresh_token. Ensure utils package is accessible.")
    check_and_refresh_token = None

# Define a constant for the threshold value (if still needed, seems unused now)
# THRESHOLD_VALUE = 10
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_SLEEP_INTERVAL = 10


# Load the config:
def load_config(config_path: Union[str, Path]) -> dict:
    """Loads configuration settings from a .toml file."""
    config_path = Path(config_path)
    logging.info("Loading configuration from: %s", config_path)
    try:
        with open(config_path, "r", encoding="utf-8") as file: # toml uses text mode
            configuration = toml.load(file)
        logging.info("Configuration loaded successfully.")
        return configuration
    except FileNotFoundError:
        logging.exception("Configuration file not found: %s", config_path)
        raise
    except toml.TomlDecodeError:
        logging.exception("Error decoding TOML file: %s", config_path)
        raise
    except Exception as e:
        logging.exception("Unexpected error loading config %s: %s", config_path, e)
        raise


def read_sic_data(file_path: Union[str, Path], config: dict) -> pd.DataFrame:
    """Reads a comma-separated CSV file and returns a DataFrame."""
    file_path = Path(file_path)
    logging.info("Reading SIC data from: %s", file_path)

    # Get column names for reading the input CSV
    input_csv_cols = config.get("column_names", {}).get("input_csv_columns")

    try:
        sic_data = pd.read_csv(
            file_path,
            delimiter=",",
            names=input_csv_cols, # Use names from config if provided
            header=0 if not input_csv_cols else None, # Assume header if names not specified
            dtype=str,
            na_filter=False,
            encoding="utf-8"
        )
        logging.info("Successfully loaded %d rows from %s", len(sic_data), file_path)
        return sic_data
    except FileNotFoundError:
        logging.exception("Data file not found: %s", file_path)
        raise
    except Exception as e:
        logging.exception("Error reading data file %s: %s", file_path, e)
        raise


def process_row(row: pd.Series, secret_code: str, config: dict) -> Optional[dict]:
    """Process a single row, make an API request, and return the response.

    Parameters:
        row (pd.Series): A row from the DataFrame.
        secret_code (str): The secret code for API authorization.
        config (dict): The application configuration dictionary.

    Returns:
        Optional[dict]: The response JSON with additional information, or None on error.
    """
    # Get column names for payload from config
    col_names_config = config.get("column_names", {})
    uid_col = col_names_config.get("payload_unique_id", "unique_id")
    job_title_col = col_names_config.get("payload_job_title", "soc2020_job_title")
    job_desc_col = col_names_config.get("payload_job_description", "soc2020_job_description")
    industry_desc_col = col_names_config.get("payload_industry_description", "sic2007_employee")

    # Get API URL from config
    api_url = config.get("api_settings", {}).get("base_url")
    if not api_url:
        logging.error("API base_url not found in configuration [api_settings]. Cannot process row.")
        return None

    # Extract data using configured column names
    unique_id = row.get(uid_col, "N/A")
    job_title = row.get(job_title_col, "")
    job_description = row.get(job_desc_col, "")
    industry_descr = row.get(industry_desc_col, "")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {secret_code}",
    }
    payload = {
        "llm": config.get("api_settings", {}).get("llm_model", "gemini"), # Example
        "type": config.get("api_settings", {}).get("request_type", "sic"), # Example
        "job_title": job_title,
        "job_description": job_description,
        "industry_descr": industry_descr,
    }

    response_json_data = None
    try:
        response = requests.post(
            api_url, headers=headers, data=json.dumps(payload), timeout=DEFAULT_REQUEST_TIMEOUT
        )
        response.raise_for_status()
        response_json_data = response.json()
        # Add metadata and payload to the successful response
        response_json_data.update({
            "unique_id": unique_id, # Ensure unique_id from input is in output
            "request_payload": payload # Include the sent payload for reference
            })
    except requests.exceptions.Timeout:
        logging.error("Request timed out for unique_id %s", unique_id)
    except requests.exceptions.HTTPError as e:
        logging.error("HTTP error for unique_id %s: %s %s", unique_id, e.response.status_code, e.response.text[:200])
    except requests.exceptions.RequestException as e:
        logging.exception("Request failed for unique_id %s: %s", unique_id, e)
    except json.JSONDecodeError:
        status = response.status_code if 'response' in locals() else 'N/A'
        text = response.text[:200] if 'response' in locals() else 'N/A'
        logging.error("Failed to decode JSON for unique_id %s. Status: %s, Text: %s", unique_id, status, text)

    # If an error occurred, response_json_data is still None.
    # If you want to return error info in the JSON, construct it here:
    if response_json_data is None:
        response_json_data = {
            "unique_id": unique_id,
            "request_payload": payload, # Still useful to log what was sent
            "error": f"Failed to process row. Check logs for unique_id {unique_id}."
        }
    return response_json_data


def process_test_set(
    secret_code: str,
    csv_filepath: Union[str, Path],
    output_filepath: Union[str, Path],
    config: dict, # Pass the loaded config
    test_mode: bool = False,
    test_limit: int = 2,
) -> None:
    """Process the test set CSV file, make API requests, and save responses."""
    csv_filepath = Path(csv_filepath)
    output_filepath = Path(output_filepath)

    logging.info("Reading input CSV for processing: %s", csv_filepath)
    try:
        # Pass config to read_sic_data if it uses it for column names
        gold_df = read_sic_data(csv_filepath, config)
    except Exception: # Catch errors from read_sic_data
        logging.error("Failed to read input CSV %s. Aborting processing.", csv_filepath)
        return

    if gold_df.empty:
        logging.warning("Input DataFrame is empty. Nothing to process.")
        return

    data_to_process = gold_df
    if test_mode:
        data_to_process = gold_df.head(test_limit)
        logging.info("Test mode enabled. Processing first %s rows.", len(data_to_process))

    logging.info("Opening output file for appending: %s", output_filepath)
    try:
        # Ensure output directory exists
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(output_filepath, "a", encoding="utf-8") as file:
            for _, row in data_to_process.iterrows():
                # Pass config to process_row
                response_json = process_row(row, secret_code, config)
                if response_json: # Only write if a response (even error dict) was generated
                    file.write(json.dumps(response_json) + "\n")
                # Consider if sleep is needed even if process_row fails
                time.sleep(DEFAULT_SLEEP_INTERVAL)
        logging.info("Finished processing. Output saved to %s", output_filepath)
    except IOError as e:
        logging.exception("Could not open or write to output file %s: %s", output_filepath, e)
    except Exception as e:
        logging.exception("An unexpected error occurred during file processing: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SIC code batch data through Survey Assist API.")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the .toml configuration file."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    main_logger = logging.getLogger(__name__)

    try:
        app_config = load_config(args.config_file)
    except Exception: # Error already logged by load_config
        main_logger.critical("Failed to load configuration. Exiting.")
        exit(1) # Use sys.exit for cleaner exit codes

    # Get paths from config
    gold_standard_csv_str = app_config.get("paths", {}).get("gold_standard_csv")
    output_file_path_str = app_config.get("paths", {}).get("output_filepath")

    if not gold_standard_csv_str or not output_file_path_str:
        main_logger.critical("Missing 'gold_standard_csv' or 'output_filepath' in config [paths]. Exiting.")
        exit(1)

    # Convert to Path objects (assuming relative to project root or where script is run)
    # For robustness, paths in config should be relative to project root,
    # and script should know project root.
    try:
        project_root = Path(__file__).resolve().parent.parent # Assumes scripts/batch.py
    except NameError:
        project_root = Path.cwd() # Fallback

    gold_standard_csv = project_root / gold_standard_csv_str
    output_file_path = project_root / output_file_path_str

    # Get parameters
    params = app_config.get("parameters", {})
    test_mode_option = params.get("test_mode", False)
    test_num = params.get("test_num", 2)


    # Get a secret token
    if check_and_refresh_token is None:
        main_logger.critical("JWT Utility not imported correctly. Exiting.")
        exit(1)

    TOKEN_START_TIME = 0
    CURRENT_TOKEN = ""
    api_gateway_env = os.getenv("API_GATEWAY") # Still needed for token generation util
    sa_email_env = os.getenv("SA_EMAIL")
    jwt_secret_path_env = os.getenv("JWT_SECRET")

    if not all([api_gateway_env, sa_email_env, jwt_secret_path_env]):
        main_logger.critical("Missing environment variables for token: API_GATEWAY, SA_EMAIL, JWT_SECRET. Exiting.")
        exit(1)

    try:
        _, CURRENT_TOKEN = check_and_refresh_token(
            TOKEN_START_TIME, CURRENT_TOKEN, jwt_secret_path_env, api_gateway_env, sa_email_env
        )
        if not CURRENT_TOKEN:
            main_logger.critical("Failed to obtain API token. Exiting.")
            exit(1)
        main_logger.info("API token obtained successfully.")
    except Exception as e:
        main_logger.exception("Error obtaining API token: %s", e)
        exit(1)

    main_logger.info("Starting batch processing...")
    process_test_set(
        secret_code=CURRENT_TOKEN,
        csv_filepath=gold_standard_csv,
        output_filepath=output_file_path,
        config=app_config, # Pass the loaded config
        test_mode=test_mode_option,
        test_limit=test_num,
    )
    main_logger.info("Batch processing finished.")

