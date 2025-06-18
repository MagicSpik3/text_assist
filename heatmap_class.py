import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any

# Assuming the rest of your class methods are also in the file
class ClassificationEvaluator:
    """
    A flexible toolkit to evaluate classification results in a DataFrame.
    """

    def __init__(self, df: pd.DataFrame, id_col: str):
        """
        Initializes the evaluator with the DataFrame to be analyzed.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("A non-empty pandas DataFrame must be provided.")
        if id_col not in df.columns:
            raise ValueError(f"The id_col '{id_col}' was not found in the DataFrame.")
        
        self.df = df.astype(str) # Ensure data is string type for reliable comparison
        self.id_col = id_col
        
    # --- Other methods like compare_columns() and evaluate_list_match() would be here ---

    def plot_confusion_heatmap(
        self,
        human_code_col: str,
        llm_code_col: str,
        top_n: int = 10,
        exclude_patterns: List[str] = None
    ) -> plt.Axes:
        """
        Generates and displays a confusion matrix heatmap for the top N codes.

        Args:
            human_code_col (str): The column name for the ground truth codes.
            llm_code_col (str): The column name for the model's predicted codes.
            top_n (int): The number of most frequent codes to include in the matrix.
            exclude_patterns (List[str]): A list of substrings to filter out from the
                                          human_code_col before analysis (e.g., ['x', '-9']).

        Returns:
            plt.Axes: The matplotlib axes object for further customization.
        """
        # --- Step 1: Create a temporary, smaller DataFrame for efficiency ---
        required_cols = [human_code_col, llm_code_col]
        if any(col not in self.df.columns for col in required_cols):
            raise ValueError("One or both specified columns not found in the DataFrame.")
            
        temp_df = self.df[required_cols].copy()

        # --- Step 2: Clean the data by excluding specified patterns ---
        if exclude_patterns:
            print(f"Initial shape before filtering: {temp_df.shape}")
            for pattern in exclude_patterns:
                temp_df = temp_df[~temp_df[human_code_col].str.contains(pattern, na=False)]
            print(f"Shape after filtering: {temp_df.shape}")
            
        # --- Step 3: Find the Most Important Codes to Display ---
        top_human_codes = temp_df[human_code_col].value_counts().nlargest(top_n).index
        top_llm_codes = temp_df[llm_code_col].value_counts().nlargest(top_n).index

        # Filter the DataFrame to only include rows with these top codes
        filtered_df = temp_df[
            temp_df[human_code_col].isin(top_human_codes) & 
            temp_df[llm_code_col].isin(top_llm_codes)
        ]
        
        if filtered_df.empty:
            print("No overlapping data found for the top codes. Cannot generate a matrix.")
            return None

        # --- Step 4: Create the Confusion Matrix ---
        confusion_matrix = pd.crosstab(
            filtered_df[human_code_col],
            filtered_df[llm_code_col]
        )

        # --- Step 5: Visualize as a Heatmap ---
        plt.figure(figsize=(12, 10))
        heatmap = sns.heatmap(
            confusion_matrix, 
            annot=True,
            fmt='d',
            cmap='YlGnBu'
        )
        
        plt.title(f'Confusion Matrix: Top {top_n} Human vs. LLM Codes', fontsize=16)
        plt.ylabel(f'Human Coder ({human_code_col})', fontsize=12)
        plt.xlabel(f'LLM Prediction ({llm_code_col})', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        return heatmap
