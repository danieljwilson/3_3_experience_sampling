import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

import xgboost as xgb

# Load data
path = '/Users/djw/Documents/pCloud_synced/Academics/Projects/2020_thesis/thesis_experiments/3_experiments/3_3_experience_sampling/3_3_2_processed_data/run_2/run2_selfReport.csv'
df = pd.read_csv(path)

# select relevant columns
df = df[['trial_date', 'sr_DAILY_past24_gap', 'ParticipantIdentifier']]
df = df.set_index('trial_date')
df.index = pd.to_datetime(df.index)

subjects = np.unique(df['ParticipantIdentifier'])

# Plot
df.loc[df['ParticipantIdentifier']==subjects[52]].plot(
    style='-', figsize=(15,5),
    color= color_pal[0],
    title='Intention Behavior Gap')
plt.show()


# Look at intention behavior gap trends
