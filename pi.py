# %% [markdown]
# ## Section 3: Pie Chart of Group Membership of Total Data
# This section calculates and displays a pie chart representing the main group
# memberships within the dataset:
# - Uncodeable to any digits
# - Codeable to two digits only
# - Codeable unambiguously to five digits
# - Codeable ambiguously to five digits
# - Any remaining data as 'Other'

# %%
# Ensure necessary libraries are available (already imported in your script)
# import matplotlib.pyplot as plt
# import seaborn as sns # For color_palette, if desired

# --- Calculate counts for each group membership category ---
N = len(eval_data)

# 1. Uncodeable to any digits
# Corresponds to "Uncodeable to any digits: 6.79 %" in your summary
count_uncodeable = (eval_data["num_answers"] == 0).sum()

# 2. Codeable to two digits only
# Corresponds to "Codeable to two digits only: 27.6 %" in your summary
# This assumes the 'Match_2_digits' column represents items codeable *only* to 2 digits
# and are mutually exclusive from 5-digit codes and uncodeable items.
count_2_digits_only = eval_data['Match_2_digits'].sum()

# 3. Codeable unambiguously to five digits
# Corresponds to "Codeable unambiguously to five digits: 59.7 %" in your summary
# This assumes 'Unambiguous' implies a 5-digit match and is exclusive of other categories.
count_5_digits_unambiguous = eval_data['Unambiguous'].sum()

# 4. Codeable ambiguously to five digits
# Corresponds to "Codeable ambiguously to five digits: 4.1 %" in your summary
# These are items that Match_5_digits but are NOT Unambiguous.
count_5_digits_ambiguous = (eval_data['Match_5_digits'] & ~eval_data['Unambiguous']).sum()

# --- Define data for the pie chart ---
group_counts_for_pie = [
    count_uncodeable,
    count_2_digits_only,
    count_5_digits_unambiguous,
    count_5_digits_ambiguous,
]

group_labels_template = [
    "Uncodeable",
    "Codeable to 2 digits only",
    "Codeable unambiguously to 5 digits",
    "Codeable ambiguously to 5 digits",
]

# Calculate percentages for precise labeling
calculated_percentages = [(count / N) * 100 for count in group_counts_for_pie]

# Create labels with percentages
chart_labels = [
    f"{label}\n({percent:.1f}%)"
    for label, percent in zip(group_labels_template, calculated_percentages)
]

# Check for 'Other' category (data not falling into the above four groups)
total_categorized_count = sum(group_counts_for_pie)
if total_categorized_count < N:
    count_other = N - total_categorized_count
    percent_other = (count_other / N) * 100
    group_counts_for_pie.append(count_other)
    chart_labels.append(f"Other\n({percent_other:.1f}%)")
elif total_categorized_count > N:
    logger.warning(
        "The sum of specified group counts exceeds total dataset size. "
        "This may indicate overlapping definitions. Pie chart might be misleading."
    )
    # Adjust counts to sum to N if strictly enforcing full pie (pro-rata or error)
    # For now, this situation suggests a data definition issue to be reviewed.
    # However, based on your summary, an 'Other' slice is expected.

# --- Create and save the pie chart ---
plt.figure(figsize=(12, 10)) # Increased size for better label readability
plt.pie(
    group_counts_for_pie,
    labels=chart_labels,
    autopct=lambda p: '{:.1f}%'.format(p) if p > 1 else '', # Show autopct % for slices > 1%
    startangle=90,
    wedgeprops={"edgecolor": "white"}, # Adds a white border to slices
    colors=sns.color_palette("pastel", len(group_counts_for_pie)) # Use a seaborn palette
)
plt.title('Distribution of Group Membership in Total Data', fontsize=16)
plt.axis('equal')  # Ensures the pie chart is circular.
plt.tight_layout() # Helps fit labels if they are long

# Save the figure
# Ensure 'output_dir_path' is defined in your script (it is, from your provided code)
output_pie_chart_path = output_dir_path / "group_membership_pie_chart.png"
plt.savefig(output_pie_chart_path)
logger.info("Pie chart of group membership saved to %s", output_pie_chart_path)

plt.show()
plt.close() # Close the figure to free memory
