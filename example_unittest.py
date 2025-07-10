import csv
import pytest
from src.transaction_processor import TransactionProcessor

# Step 1: Create a function to load your test data from the CSV
def load_test_data(filename="tests/test_data.csv"):
    """Loads test data from a CSV file."""
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield row

# Step 2: Use pytest.mark.parametrize to create a test for each row
@pytest.mark.parametrize("test_case", load_test_data())
def test_categorize_transaction(test_case):
    """
    Tests the categorize_transaction method with data from the CSV.
    Each row in the CSV will be run as a separate test.
    """
    # ARRANGE: Set up the test with data from the current CSV row
    processor = TransactionProcessor()
    description = test_case['description']
    expected_category = test_case['expected_category']

    # ACT: Call the method you want to test
    actual_category = processor.categorize_transaction(description)

    # ASSERT: Check if the result is what you expect
    assert actual_category == expected_category
