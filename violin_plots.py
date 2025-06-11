import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
# Path to the data file that contains at least the following columns:
#   1. "PID" – unique identifier of each participant
#   2. "emotional_intelligence" – emotional-intelligence scale score
#   3. "battery_outcome_rw_intention_behavior_gap_avg" – intention behaviour gap
# Adjust the file name or column names below if they differ in your dataset.
DATA_PATH = "your_dataset.csv"  # <-- replace with the real path

# Column names (feel free to change if yours differ)
COL_PID = "PID"
COL_EI = "emotional_intelligence"
COL_IB_GAP = "battery_outcome_rw_intention_behavior_gap_avg"

JITTER = 0.15  # horizontal jitter range (+/-)

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

# -----------------------------------------------------------------------------
# BASIC SANITY CHECKS
# -----------------------------------------------------------------------------
missing_cols = {COL_PID, COL_EI, COL_IB_GAP} - set(df.columns)
if missing_cols:
    raise ValueError(f"The following required columns are missing in the dataset: {missing_cols}")

# Pre-compute a unique jitter for each participant on each axis so that
# 1) dots are horizontally separated and 2) connecting lines terminate
#    exactly on the dots.
df_plot = df.copy()  # work on a copy so we can add helper columns safely
# Skip rows with missing values first
mask_complete = df_plot[[COL_EI, COL_IB_GAP]].notna().all(axis=1)
df_plot = df_plot.loc[mask_complete].copy()

rng = np.random.default_rng(seed=42)  # reproducible jitter
# Jitter for EI dots (left)
df_plot["_jx_ei"] = rng.uniform(-JITTER, JITTER, len(df_plot))
# Jitter for Gap dots (right)
df_plot["_jx_gap"] = rng.uniform(-JITTER, JITTER, len(df_plot))

# -----------------------------------------------------------------------------
# SELECT THREE PARTICIPANTS TO HIGHLIGHT
#   • consider only rows where the intention-behaviour gap is within [40, 50]
#   • pick:
#       1) PID with the highest EI
#       2) PID with the lowest EI
#       3) PID whose EI is closest to the mean EI of that sub-group
#     (duplicates are removed in that priority order)
# -----------------------------------------------------------------------------

# 1. Narrow down to the requested gap range
eligible = df_plot[(df_plot[COL_IB_GAP] >= 40) & (df_plot[COL_IB_GAP] <= 50)].copy()

# If there are no eligible rows we won't highlight anything
highlight_ordered: list[str] = []

if not eligible.empty:
    # Highest EI
    pid_max = eligible.loc[eligible[COL_EI].idxmax(), COL_PID]
    highlight_ordered.append(pid_max)

    # Lowest EI
    pid_min = eligible.loc[eligible[COL_EI].idxmin(), COL_PID]
    if pid_min not in highlight_ordered:
        highlight_ordered.append(pid_min)

    # EI closest to mean
    mean_ei = eligible[COL_EI].mean()
    pid_mean = (eligible.assign(abs_diff=(eligible[COL_EI] - mean_ei).abs())
                        .sort_values("abs_diff")
                        .iloc[0][COL_PID])
    if pid_mean not in highlight_ordered:
        highlight_ordered.append(pid_mean)

# Restrict to at most 3
highlight_sample = highlight_ordered[:3]

# Assign each highlighted PID a distinct colour
highlight_palette = sns.color_palette("tab10", n_colors=len(highlight_sample))
highlight_colours = dict(zip(highlight_sample, highlight_palette))

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

sns.set_theme(style="whitegrid")

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(10, 6),
    constrained_layout=True,
    dpi=110,
)

# ------------------------- 1) Emotional Intelligence -------------------------
sns.violinplot(
    y=df_plot[COL_EI],
    ax=ax1,
    palette=["#7DC9E7"],
    inner=None,
    cut=0,
    linewidth=1,
)
# Draw jittered dots manually so we control their x-coordinate
ax1.scatter(
    df_plot["_jx_ei"],
    df_plot[COL_EI],
    color="black",
    s=20,
    zorder=3,
    alpha=0.8,
)
ax1.set_title("Emotional Intelligence")
ax1.set_xlabel("")
ax1.set_ylabel("Score")
ax1.set_xticks([])

# --------------------- 2) Intention–Behaviour Gap ----------------------------
sns.violinplot(
    y=df_plot[COL_IB_GAP],
    ax=ax2,
    palette=["#AACF91"],
    inner=None,
    cut=0,
    linewidth=1,
)
ax2.scatter(
    df_plot["_jx_gap"],
    df_plot[COL_IB_GAP],
    color="black",
    s=20,
    zorder=3,
    alpha=0.8,
)
ax2.set_title("Intention–Behaviour Gap")
ax2.set_xlabel("")
ax2.set_ylabel("Gap (avg)")
# Move y-axis to the right for ax2
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_xticks([])

# -------------------- 3) Connect participants across axes -------------------
for _, row in df_plot.iterrows():
    pid = row[COL_PID]
    # Determine styling depending on whether this PID is highlighted
    if pid in highlight_colours:
        line_colour = highlight_colours[pid]
        lw = 2.2  # bold
        alpha = 1.0
    else:
        line_colour = "grey"
        lw = 0.7
        alpha = 0.4
    connection = ConnectionPatch(
        xyA=(row["_jx_ei"], row[COL_EI]),
        coordsA="data",
        axesA=ax1,
        xyB=(row["_jx_gap"], row[COL_IB_GAP]),
        coordsB="data",
        axesB=ax2,
        color=line_colour,
        alpha=alpha,
        linewidth=lw,
        zorder=2 if pid in highlight_colours else 1,
    )
    connection.set_in_layout(False)
    fig.add_artist(connection)

# -------------------- 4) Overlay highlighted dots ---------------------------
for pid, colour in highlight_colours.items():
    subset = df_plot[df_plot[COL_PID] == pid]
    # EI dot
    ax1.scatter(
        subset["_jx_ei"],
        subset[COL_EI],
        color=colour,
        s=50,
        zorder=4,
        edgecolor="white",
        linewidth=0.8,
    )
    # Gap dot
    ax2.scatter(
        subset["_jx_gap"],
        subset[COL_IB_GAP],
        color=colour,
        s=50,
        zorder=4,
        edgecolor="white",
        linewidth=0.8,
    )

# -----------------------------------------------------------------------------
# SHOW / SAVE FIGURE
# -----------------------------------------------------------------------------
# Uncomment the next line if you want to save directly to a file.
# fig.savefig("violin_plots_joined.png", bbox_inches="tight")

plt.show() 