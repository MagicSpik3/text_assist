from typing import Dict, Any

class LabelAccuracy:
    # ... (your __init__ and other methods remain the same) ...

    def get_accuracy(self, threshold: float = 0.0, match_type: str = "full") -> Dict[str, Any]:
        """
        Calculate accuracy for predictions above a confidence threshold.

        Args:
            threshold (float): Minimum confidence score threshold.
            match_type (str): The type of accuracy to calculate.
                              Options: 'full' (default) or '2-digit'.

        Returns:
            Dict[str, Any]: A dictionary containing the accuracy percentage,
                            match count, non-match count, and total.
        """
        if match_type == "2-digit":
            correct_col = "is_correct_2_digit"
        elif match_type == "full":
            correct_col = "is_correct"
        else:
            raise ValueError("match_type must be 'full' or '2-digit'")

        if correct_col not in self.df.columns:
            raise RuntimeError(
                f"Derived column '{correct_col}' not found. Ensure _add_derived_columns ran."
            )

        # 1. Filter the DataFrame based on the confidence threshold
        filtered_df = self.df[self.df["max_score"] >= threshold]
        total_in_subset = len(filtered_df)

        # Handle the edge case where no data meets the threshold
        if total_in_subset == 0:
            return {
                "accuracy_percent": 0.0,
                "matches": 0,
                "non_matches": 0,
                "total_considered": 0,
            }

        # 2. Calculate the raw counts
        # .sum() on a boolean column counts the number of True values
        match_count = filtered_df[correct_col].sum()
        non_match_count = total_in_subset - match_count
        
        # 3. Calculate the percentage
        accuracy_percent = (100 * match_count / total_in_subset)

        # 4. Return all values in a structured dictionary
        return {
            "accuracy_percent": round(accuracy_percent, 1),
            "matches": int(match_count),
            "non_matches": int(non_match_count),
            "total_considered": total_in_subset,
        }
