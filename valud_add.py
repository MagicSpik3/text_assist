from typing import Dict, Any

class LabelAccuracy:
    # ... your existing __init__, get_accuracy, etc. methods are here ...

    def get_candidate_contribution(self, candidate_col: str) -> Dict[str, Any]:
        """
        Assesses the value add of a single candidate column.

        This method calculates how often a specific candidate's prediction matches
        any of the ground truth labels, and also how often it specifically
        matches the primary ground truth label.

        Args:
            candidate_col (str): The name of the candidate column to evaluate
                                 (e.g., 'candidate_5_sic_code').

        Returns:
            Dict[str, Any]: A dictionary containing the match rates and counts.
        """
        # --- 1. Validate that all necessary columns exist ---
        primary_clerical_col = self.clerical_label_cols[0]
        required_cols = [candidate_col, primary_clerical_col] + self.clerical_label_cols
        
        if any(col not in self.df.columns for col in required_cols):
            raise ValueError(f"One or more required columns not found for this analysis.")

        # Create a working copy, dropping rows where the candidate has no code
        # This ensures we only evaluate the candidate where it made a prediction.
        working_df = self.df[[candidate_col, *self.clerical_label_cols]].dropna(
            subset=[candidate_col]
        )
        total_considered = len(working_df)

        if total_considered == 0:
            return {
                "candidate_column": candidate_col,
                "total_predictions_made": 0,
                "primary_match_percent": 0.0,
                "primary_match_count": 0,
                "any_clerical_match_percent": 0.0,
                "any_clerical_match_count": 0,
            }

        # --- 2. Calculate match against the PRIMARY clerical code ---
        primary_matches_mask = (
            working_df[candidate_col] == working_df[primary_clerical_col]
        )
        primary_match_count = primary_matches_mask.sum()
        primary_match_percent = 100 * primary_match_count / total_considered

        # --- 3. Calculate match against ANY of the clerical codes ---
        def check_row_for_any_match(row):
            # Get the single value from the candidate column for this row
            candidate_value = row[candidate_col]
            # Get the list of all clerical codes for this row
            clerical_values = list(row[self.clerical_label_cols].dropna())
            return candidate_value in clerical_values

        any_match_mask = working_df.apply(check_row_for_any_match, axis=1)
        any_match_count = any_match_mask.sum()
        any_match_percent = 100 * any_match_count / total_considered

        # --- 4. Return results in a structured dictionary ---
        return {
            "candidate_column": candidate_col,
            "total_predictions_made": total_considered,
            "primary_match_percent": round(primary_match_percent, 2),
            "primary_match_count": int(primary_match_count),
            "any_clerical_match_percent": round(any_match_percent, 2),
            "any_clerical_match_count": int(any_match_count),
        }
