# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: survey-assist-utils-PWI-TvqZ-py3.12
#     language: python
#     name: python3
# ---

# %%
"""Summary analysis of the TLFS_evaluation_data_IT2 dataset."""
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

# %%
# --- Load Data and Define Constants ---
# Assuming these are defined in previous cells as per your original code
output_dir_path = Path("/home/user/survey-assist-utils/notebooks")
eval_data = pd.read_csv(
    "../data/analysis_outputs/TLFS_evaluation_data_IT2_output.csv",
    dtype={"SIC_Division": str},
)
X_COUNT_FOR_MULTI_ANSWERS = 4

# %%
# --- Section 1: Group Membership of Total Data (Refactored for Styling) ---

# --- Calculate counts for each group membership category ---
N = len(eval_data)

# 1. Uncodeable to any digits
count_uncodeable = (eval_data["num_answers"] == 0).sum()

# 2. Codeable to two digits only
count_2_digits_only = eval_data["Match_2_digits"].sum()

# 3. Codeable unambiguously to five digits
count_5_digits_unambiguous = eval_data["Unambiguous"].sum()

# 4. Codeable ambiguously to five digits
count_5_digits_ambiguous = (
    eval_data["Match_5_digits"] & ~eval_data["Unambiguous"]
).sum()

# 5. Other categories
count_3_digits_only = eval_data["Match_3_digits"].sum()
count_four_plus = (eval_data["num_answers"] == X_COUNT_FOR_MULTI_ANSWERS).sum()

# --- Define data for the pie chart ---
group_counts_for_pie = [
    count_uncodeable,
    count_four_plus,
    count_2_digits_only,
    count_3_digits_only,
    count_5_digits_unambiguous,
    count_5_digits_ambiguous,
]

group_labels_template = [
    "Uncodeable",
    "4+ Codes",
    "Codeable to 2 digits only",
    "Codeable to 3 digits only",
    "Codeable unambiguously to 5 digits",
    "Codeable ambiguously to 5 digits",
]

# --- Handle 'Other' category ---
total_categorised_count = sum(group_counts_for_pie)
count_other = N - total_categorised_count
if count_other > 0:
    group_counts_for_pie.append(count_other)
    group_labels_template.append("Other")

# --- Styling Refactor ---

# 1. Define the official GSS colour palette from the guidance
# Source: https://analysisfunction.civilservice.gov.uk/policy-store/data-visualisation-colours-in-charts/
gss_palette = [
    "#12436D",  # Main Blue
    "#28A197",  # Main Teal
    "#801650",  # Main Magenta
    "#F46A25",  # Main Orange
    "#3D3D3D",  # Dark Grey
    "#A285D1",  # Supporting Purple
    "#707070",  # Medium Grey (for the 'Other' category if needed)
]

# Calculate percentages for precise labeling
calculated_percentages = [(count / N) * 100 for count in group_counts_for_pie]

# Create labels with percentages
chart_labels = [
    f"{label}\n({percent:.1f}%)"
    for label, percent in zip(group_labels_template, calculated_percentages)
]

# --- Create and save the styled pie chart ---
plt.figure(figsize=(12, 12))  # Increased figure size for better label spacing
plt.pie(
    group_counts_for_pie,
    labels=chart_labels,
    autopct=lambda p: f"{p:.1f}%" if p >= 2 else "",  # Show autopct for slices >= 2%
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    # 2. Apply the official GSS palette
    colors=gss_palette[:len(group_counts_for_pie)],
    # 3. Increase font size for better readability
    textprops={'fontsize': 14}
)

plt.title("Distribution of Group Membership in Total Data", fontsize=20, weight='bold')
plt.axis("equal") # Ensures the pie chart is a circle.
plt.tight_layout()

# Save the figure
output_pie_chart_path = output_dir_path / "group_membership_pie_chart_styled.png"
plt.savefig(output_pie_chart_path, bbox_inches='tight')

plt.show()
plt.close()

# --- Create and display the summary table (Unchanged) ---
group_labels_with_total = group_labels_template + ["Total"]
group_counts_with_total = group_counts_for_pie + [sum(group_counts_for_pie)]
percentages_with_total = calculated_percentages + [sum(calculated_percentages)]

df_summary = pd.DataFrame(
    {
        "Category": group_labels_with_total,
        "Count": group_counts_with_total,
        "Percentage": [f"{p:.1f}%" for p in percentages_with_total],
    }
)
print("\n--- Summary Table ---")
print(df_summary.to_string())
