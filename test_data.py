import pandas as pd
import io

csv_data = """unique_id,CC_1,CC_2,CC_3,SA_1,SA_2,SA_3,SA_4,SA_5,SA_score_1,SA_score_2,SA_score_3,SA_score_4,SA_score_5
test_001,11111,22222,33333,44444,55555,11199,66666,77777,0.9,0.8,0.7,0.6,0.5
test_002,10001,20002,30003,40004,50005,20002,60006,30099,0.8,0.7,0.9,0.6,0.5
test_003,12345,67890,54321,54321,99999,88888,12345,77777,0.9,0.6,0.5,0.8,0.4
test_004,77700,88800,99900,11100,88800,99900,22200,77700,0.5,0.9,0.8,0.4,0.7
test_005,25000,,75000,25000,35000,,45000,75099,0.9,0.8,0.7,0.6,0.5
"""

# Define the data types to ensure 5-digit codes are treated as text (strings)
string_cols = {
    'CC_1': str, 'CC_2': str, 'CC_3': str,
    'SA_1': str, 'SA_2': str, 'SA_3': str, 'SA_4': str, 'SA_5': str
}

# Read the data into a pandas DataFrame
df = pd.read_csv(io.StringIO(csv_data), dtype=string_cols)

print("Test data loaded successfully!")
print(df)
print("\nData types for CC columns:")
print(df[['CC_1', 'CC_2', 'CC_3']].info())
