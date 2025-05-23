import pandas as pd
import json

def flatten_llm_json_to_dataframe(file_path: str, max_candidates: int = 5) -> pd.DataFrame:
    """
    Reads LLM response JSON data from a file, flattens it into a Pandas DataFrame.

    Args:
        file_path (str): The path to the JSON file.
        max_candidates (int): The maximum number of SIC candidates to flatten per record.

    Returns:
        pd.DataFrame: A Pandas DataFrame with the flattened JSON data.
    """
    all_flat_records = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return pd.DataFrame()

    # Ensure json_data is a list of records
    if isinstance(json_data, dict):
        records_to_process = [json_data] # Handle case where file contains a single JSON object
    elif isinstance(json_data, list):
        records_to_process = json_data
    else:
        print(f"Error: JSON content is not a list or a single object (dictionary).")
        return pd.DataFrame()

    for record in records_to_process:
        flat_record = {}

        # 1. Add top-level fields
        flat_record['unique_id'] = record.get('unique_id')
        flat_record['classified'] = record.get('classified')
        flat_record['followup'] = record.get('followup')
        # Rename top-level sic_code & sic_description to avoid clashes with candidate fields
        flat_record['chosen_sic_code'] = record.get('sic_code')
        flat_record['chosen_sic_description'] = record.get('sic_description')
        flat_record['reasoning'] = record.get('reasoning')

        # 2. Flatten request_payload
        payload = record.get('request_payload', {}) # Default to empty dict if payload is missing
        flat_record['payload_llm'] = payload.get('llm')
        flat_record['payload_type'] = payload.get('type')
        flat_record['payload_job_title'] = payload.get('job_title')
        flat_record['payload_job_description'] = payload.get('job_description')
        flat_record['payload_industry_descr'] = payload.get('industry_descr')

        # 3. Flatten sic_candidates
        candidates = record.get('sic_candidates', []) # Default to empty list
        for i in range(max_candidates):
            if i < len(candidates) and isinstance(candidates[i], dict):
                candidate = candidates[i]
                flat_record[f'candidate_{i+1}_sic_code'] = candidate.get('sic_code')
                flat_record[f'candidate_{i+1}_sic_descriptive'] = candidate.get('sic_descriptive')
                flat_record[f'candidate_{i+1}_likelihood'] = candidate.get('likelihood')
            else:
                # Fill with None if fewer than max_candidates or candidate data is malformed
                flat_record[f'candidate_{i+1}_sic_code'] = None
                flat_record[f'candidate_{i+1}_sic_descriptive'] = None
                flat_record[f'candidate_{i+1}_likelihood'] = None
        
        all_flat_records.append(flat_record)

    df = pd.DataFrame(all_flat_records)

    # Optionally, set 'unique_id' as index if it exists and all values are unique
    if 'unique_id' in df.columns and df['unique_id'].nunique() == len(df):
        df = df.set_index('unique_id')
    elif 'unique_id' in df.columns:
        print("Warning: 'unique_id' column contains duplicate values or NaNs, not set as index.")


    return df

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy JSON file with content similar to your example
    # Your example string is a list containing one dictionary. We'll add another for demonstration.
    example_json_content = """
[
    {
        "classified": false,
        "followup": "Does the company provide quality improvement services to hospitals or other healthcare providers?",
        "sic_code": "74909",
        "sic_description": "Other professional, scientific and technical activities (not including environmental consultancy or quantity surveying) nec",
        "sic_candidates": [
            {
                "sic_code": "74909",
                "sic_descriptive": "Other professional, scientific and technical activities (not including environmental consultancy or quantity surveying) nec",
                "likelihood": 0.7
            },
            {
                "sic_code": "86101",
                "sic_descriptive": "Hospital activities",
                "likelihood": 0.3
            }
        ],
        "reasoning": "The job title QUALITY IMPROVEMENT LEAD and job description LEADING ON ENSURING THAT QUALITY CLINICAL CARE IS DELIVERED TO PATIENTS suggest activities related to quality assurance and potentially healthcare. Code 74909, encompassing Quality assurance consultancy activities, is a plausible match given the focus on quality improvement. Code 86101, Hospital activities, is considered a less likely alternative, as the respondent's data describes a role within quality improvement rather than direct hospital operation. A follow-up question is needed to clarify whether the company's services are directly provided to healthcare facilities or if they are a consultancy providing quality improvement services to various clients, including healthcare.",
        "unique_id": "EV000017",
        "request_payload": {
            "llm": "gemini",
            "type": "sic",
            "job_title": "QUALITY IMPROVEMENT LEAD",
            "job_description": "LEADING ON ENSURING THAT QUALITY CLINICAL CARE IS DELIVERED TO PATOENTS",
            "industry_descr": "A LARGE HOSPITAL NHS"
        }
    },
    {
        "classified": true,
        "followup": null,
        "sic_code": "12345",
        "sic_description": "Specific Widget Manufacturing",
        "sic_candidates": [
            {
                "sic_code": "12345",
                "sic_descriptive": "Specific Widget Manufacturing",
                "likelihood": 0.99
            }
        ],
        "reasoning": "Clear from description.",
        "unique_id": "EV000018",
        "request_payload": {
            "llm": "gemini",
            "type": "sic",
            "job_title": "WIDGET MAKER",
            "job_description": "MAKES WIDGETS",
            "industry_descr": "MANUFACTURING"
        }
    }
]
    """
    dummy_file_name = "llm_output_data.json"
    with open(dummy_file_name, 'w', encoding='utf-8') as f:
        f.write(example_json_content)

    # Flatten the JSON data from the file
    df_llm = flatten_llm_json_to_dataframe(dummy_file_name, max_candidates=5)

    # Print the DataFrame
    if not df_llm.empty:
        print("Flattened LLM Response DataFrame:")
        # To display all columns if the DataFrame is wide:
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        # print(df_llm)
        print(df_llm.head().to_string()) # Print first few rows, full width

        # Example of accessing a specific record by unique_id (if set as index)
        if "EV000017" in df_llm.index:
             print("\nAccessing record EV000017:")
             print(df_llm.loc["EV000017"]['chosen_sic_code'])
             print(df_llm.loc["EV000017"]['candidate_1_sic_code'])
             print(df_llm.loc["EV000017"]['candidate_2_sic_descriptive']) # Will be None for EV000017
             if 'candidate_3_sic_code' in df_llm.columns: # Check if column exists
                 print(df_llm.loc["EV000017"]['candidate_3_sic_code']) # Will be None
    else:
        print("DataFrame is empty. Check for errors.")

    # Clean up the dummy file (optional)
    # import os
    # os.remove(dummy_file_name)
