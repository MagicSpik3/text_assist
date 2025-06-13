import pandas as pd
import pytest

# Assuming your SicAnalysis class is in a file named `my_analyzer.py`
# If it's in the same file, you don't need the 'from ... import' line.
from my_analyzer import SicAnalysis 

# The `tmp_path` argument is a special pytest "fixture" that provides
# a temporary directory for our test file, so we don't clutter our project.
def test_sic_analysis_calculations(tmp_path):
    """
    Unit test for the SicAnalysis class.
    It follows the "Arrange, Act, Assert" pattern.
    """
    # 1. ARRANGE: Set up our test data and environment.
    #----------------------------------------------------
    
    # Create a small, predictable DataFrame for the test.
    test_data = pd.DataFrame({
        "sic_ind_occ1":    ["1234", "5678", "1299", "8800"],
        "chosen_sic_code": ["1234", "9999", "1234", "7700"]
    })

    # Create a path to a temporary CSV file inside the temp directory.
    temp_csv_path = tmp_path / "test_data.csv"
    
    # Save our test DataFrame to that temporary CSV file.
    test_data.to_csv(temp_csv_path, index=False)

    # Expected results from our 4 rows of test data:
    # - Full match: 1 out of 4 rows ("1234" == "1234") -> 25.0%
    # - 2-digit match: 2 out of 4 rows ('12'=='12', '12'=='12') -> 50.0%
    expected_full_match_rate = 25.0
    expected_2_digit_match_rate = 50.0

    # 2. ACT: Run the code we are testing.
    #-----------------------------------------
    
    # Initialize our analyzer with the path to the temporary CSV.
    analyzer = SicAnalysis(filepath=temp_csv_path)

    # Call the methods to get the actual results.
    actual_full_match_rate = analyzer.calculate_first_choice_rate()
    actual_2_digit_match_rate = analyzer.calculate_match_rate_at_n(2)
    

    # 3. ASSERT: Check if the results are what we expect.
    #-----------------------------------------------------
    
    assert actual_full_match_rate == expected_full_match_rate
    assert actual_2_digit_match_rate == expected_2_digit_match_rate
