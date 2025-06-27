import pandas as pd
import numpy as np
from typing import Set, Any

# Inside your LabelAccuracy class:

    def _get_sets_from_row(self, row: pd.Series) -> tuple[Set[Any], Set[Any]]:
        """
        A private helper to extract unique, non-null sets of labels from a row.

        Args:
            row (pd.Series): A single row of the DataFrame.

        Returns:
            tuple[Set[Any], Set[Any]]: A tuple containing two sets:
                                       the model labels and the clerical labels.
        """
        # Get unique, non-null values from the model prediction columns
        model_labels = set(pd.Series(row[self.model_label_cols]).dropna().unique())
        
        # Get unique, non-null values from the ground truth columns
        clerical_labels = set(pd.Series(row[self.clerical_label_cols]).dropna().unique())
        
        return model_labels, clerical_labels

    def get_jaccard_similarity(self) -> float:
        """
        Calculates the Jaccard Similarity Index for each row and returns the average.

        For each row, it compares the set of model-predicted labels with the set
        of clerical (ground truth) labels. The Jaccard Index is the size of
        the intersection divided by the size of the union of these two sets.

        Returns:
            float: The average Jaccard Similarity Index across all rows in the
                   DataFrame, returned as a percentage.
        """
        
        def calculate_jaccard_for_row(row):
            model_set, clerical_set = self._get_sets_from_row(row)

            # If either set is empty, the union will not be empty if the other
            # set has items. An empty intersection results in a score of 0.
            if not model_set and not clerical_set:
                return 1.0  # Both sets are empty, perfect agreement.
            
            intersection_size = len(model_set.intersection(clerical_set))
            union_size = len(model_set.union(clerical_set))
            
            if union_size == 0:
                return 1.0 # Technically covered by the check above, but safe.
                
            return intersection_size / union_size

        # Apply the calculation to every row of the DataFrame
        jaccard_scores = self.df.apply(calculate_jaccard_for_row, axis=1)
        
        # Return the average score across all rows, as a percentage
        average_jaccard_percentage = jaccard_scores.mean() * 100
        
        return round(average_jaccard_percentage, 2)
