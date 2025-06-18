import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Step 1: Prepare Your Data ---
# For this example, I'll create a sample DataFrame.
# In your real code, this would be your `merged_df` from your pipeline.
# It MUST have one column for the human's code and one for the LLM's top choice.

# Let's assume these are the two columns we care about:
human_code_col = 'CC_1'      # The ground truth from the human coder
llm_code_col = 'SA_1'      # The top prediction from the LLM

# Sample Data:
# In this data, the LLM is good at '11111' but often confuses '86210' with '86900'.
data = {
    'unique_id': [f'id_{i}' for i in range(100)],
    human_code_col: 
        ['11111'] * 40 +  # 40 cases of 11111
        ['86210'] * 30 +  # 30 cases of 86210
        ['45110'] * 20 +  # 20 cases of 45110
        ['99999'] * 10,   # 10 cases of a rare code
    llm_code_col: 
        ['11111'] * 35 + ['11112'] * 5 +                                 # Mostly correct, some confusion
        ['86210'] * 15 + ['86900'] * 10 + ['86210'] * 5 +                 # Often confuses 86210 with 86900
        ['45110'] * 12 + ['45190'] * 8 +                                 # Some confusion
        ['12345'] * 10                                                  # Always wrong for this rare code
}
df = pd.DataFrame(data)


# --- Step 2: Find the Most Important Codes to Display ---
# This keeps the final matrix clean and focused.

# Define how many "top" codes you want to see. Let's start with the top 5.
N = 5 

# Get the most frequent codes from the human coder column
top_human_codes = df[human_code_col].value_counts().nlargest(N).index
print(f"Top {N} Human Codes:\n{top_human_codes.tolist()}\n")

# Get the most frequent codes from the LLM's prediction column
top_llm_codes = df[llm_code_col].value_counts().nlargest(N).index
print(f"Top {N} LLM Codes:\n{top_llm_codes.tolist()}\n")

# Create a new DataFrame that ONLY includes rows where the codes are in our top lists.
filtered_df = df[
    df[human_code_col].isin(top_human_codes) & 
    df[llm_code_col].isin(top_llm_codes)
]


# --- Step 3: Create the Confusion Matrix ---
# A confusion matrix is just a frequency table. pandas.crosstab is perfect for this.
# It counts how many times each pair of (Human, LLM) codes appears.
confusion_matrix = pd.crosstab(
    filtered_df[human_code_col],
    filtered_df[llm_code_col]
)

print("--- Confusion Matrix ---")
print(confusion_matrix)


# --- Step 4: Visualize as a Heatmap ---
# A heatmap makes the numbers in the matrix easy to interpret visually.
plt.figure(figsize=(10, 8)) # Set the figure size to make it readable

heatmap = sns.heatmap(
    confusion_matrix, 
    annot=True,     # This writes the numbers inside the squares
    fmt='d',        # Format the numbers as integers
    cmap='YlGnBu'   # A nice color scheme: Yellow -> Green -> Blue
)

plt.title(f'Confusion Matrix: Top {N} Human vs. LLM Codes', fontsize=16)
plt.ylabel('Human Coder (Ground Truth)', fontsize=12)
plt.xlabel('LLM Prediction', fontsize=12)
plt.tight_layout() # Adjust layout to make sure everything fits
plt.show()
