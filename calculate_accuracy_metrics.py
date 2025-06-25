def calculate_analysis_metrics(
    analyzer: LabelAccuracy, config: ColumnConfig
) -> Dict[str, Any]:
    """
    Runs a suite of analyses and returns the results in a dictionary,
    including detailed match/non-match counts.

    Args:
        analyzer: An initialized LabelAccuracy instance.
        config: The configuration object used for the analysis.

    Returns:
        A dictionary containing all calculated metrics for one row of the results matrix.
    """
    # 1. Call the updated get_accuracy method for both match types
    full_acc_stats = analyzer.get_accuracy(match_type="full")
    digit_acc_stats = analyzer.get_accuracy(match_type="2-digit")
    
    # Get the overall coverage (this can remain as is)
    coverage = analyzer.get_coverage()
    
    # Get the detailed summary stats dictionary
    summary_stats = analyzer.get_summary_stats()

    # 2. Combine all results into a single, detailed dictionary
    # Unpack the dictionaries from get_accuracy with descriptive prefixes
    results = {
        # --- Full Match Stats ---
        "accuracy_full_match_percent": full_acc_stats["accuracy_percent"],
        "full_match_count": full_acc_stats["matches"],
        "full_match_unmatched": full_acc_stats["non_matches"],
        "full_match_total_considered": full_acc_stats["total_considered"],
        
        # --- 2-Digit Match Stats ---
        "accuracy_2_digit_match_percent": digit_acc_stats["accuracy_percent"],
        "2_digit_match_count": digit_acc_stats["matches"],
        "2_digit_match_unmatched": digit_acc_stats["non_matches"],
        "2_digit_match_total_considered": digit_acc_stats["total_considered"],
        
        # --- Other Stats ---
        "overall_coverage_percent": f"{coverage:.1f}",
        **summary_stats  # Unpack the rest of the summary stats
    }
    return results
