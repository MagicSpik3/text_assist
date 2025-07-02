"""
A helper script for generating presentation-quality charts directly from a
Jupyter Notebook, using a standardized, professional style.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import List

# --- Step 1: The Centralized Styling Function ---

def set_presentation_style(font_scale: float = 1.2):
    """
    Applies a standardized, presentation-ready style to all subsequent plots.
    This uses the GSS colour palette and Arial font for a professional look.

    Args:
        font_scale (float): Multiplier for font sizes. Adjust as needed.
    """
    # Define the official GSS colour palette
    gss_palette = [
        "#12436D",  # Main Blue
        "#28A197",  # Main Teal
        "#801650",  # Main Magenta
        "#F46A25",  # Main Orange
        "#3D3D3D",  # Dark Grey
        "#A285D1",  # Supporting Purple
        "#707070",  # Medium Grey
    ]
    sns.set_palette(gss_palette)

    # Set the context and global font properties
    sns.set_context("talk", font_scale=font_scale)
    plt.rcParams.update({
        # Use Arial, which should now be available in your VM
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        
        # Set default text and label colors for consistency
        "text.color": "#3D3D3D",
        "axes.labelcolor": "#3D3D3D",
        "xtick.color": "#3D3D3D",
        "ytick.color": "#3D3D3D",
        
        # Remove unnecessary chart borders ("spines")
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    print("Presentation plot style applied.")


# --- Step 2: A Dedicated Function to Create a Specific Chart ---

def create_group_membership_pie_chart(
    df: pd.DataFrame, output_dir: Path
) -> None:
    """
    Calculates group membership stats and saves a presentation-quality pie chart.

    Args:
        df (pd.DataFrame): The input DataFrame containing the necessary columns.
        output_dir (Path): The directory where the final chart image will be saved.
    """
    # --- A. Data Preparation (Same as your original code) ---
    N = len(df)
    group_counts = {
        "Uncodeable": (df["num_answers"] == 0).sum(),
        "4+ Codes": (df["num_answers"] == 4).sum(),
        "Codeable to 2 digits only": df["Match_2_digits"].sum(),
        "Codeable to 3 digits only": df["Match_3_digits"].sum(),
        "Codeable unambiguously to 5 digits": df["Unambiguous"].sum(),
        "Codeable ambiguously to 5 digits": (
            df["Match_5_digits"] & ~df["Unambiguous"]
        ).sum(),
    }
    total_categorised = sum(group_counts.values())
    group_counts["Other"] = N - total_categorised
    
    # Filter out zero-count categories to avoid clutter
    data_for_pie = {k: v for k, v in group_counts.items() if v > 0}
    
    chart_labels = [
        f"{label}\n({(count / N) * 100:.1f}%)"
        for label, count in data_for_pie.items()
    ]
    
    # --- B. Chart Creation and Styling ---
    
    # Apply our master style before creating the plot
    set_presentation_style(font_scale=0.8) # Use a smaller scale for pie chart labels

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    
    wedges, texts, autotexts = ax.pie(
        data_for_pie.values(),
        labels=chart_labels,
        autopct="%1.1f%%", # Add percentages inside slices
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        pctdistance=0.85 # Move percentage text inside the slice
    )

    # --- C. Fine-tuning for Presentation Quality ---
    
    # Set title with specific font properties
    ax.set_title(
        "Over 60% of Cases are Unambiguously Coded",
        fontsize=20,
        fontweight='bold',
        pad=20 # Add some padding below the title
    )
    
    # Make the percentage text inside the slices bold and white
    plt.setp(autotexts, size=12, weight="bold", color="white")
    
    # Ensure labels outside are readable
    plt.setp(texts, size=14)
    
    ax.axis('equal')

    # --- D. Save the Final Image ---
    # This is the crucial step. bbox_inches='tight' removes excess whitespace.
    output_path = output_dir / "group_membership_pie_chart.png"
    plt.savefig(output_path, bbox_inches='tight', transparent=True)
    print(f"Chart saved to: {output_path}")
    
    plt.show() # Display in notebook
    plt.close(fig) # Close the figure to free up memory


# --- Example Usage in Your Notebook ---

# Assuming 'eval_data' is your loaded DataFrame
# and 'output_dir_path' is a Path object for your output folder.

# Create a dummy DataFrame for demonstration if needed
# eval_data = pd.DataFrame(...) 
# output_dir_path = Path("./presentation_charts")
# output_dir_path.mkdir(exist_ok=True)

# create_group_membership_pie_chart(df=eval_data, output_dir=output_dir_path)
