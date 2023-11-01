import pandas as pd
import numpy as np

def all_dates_subjects_df(run_num, subjects):
    """
    Returns an empty dataframe just with all subjects
    and trial_dates included.
    """
    # Create a series of dates from '2022-09-27' to '2022-12-20'
    if run_num == 1:
        date_series = pd.date_range(start='2022-09-27', end='2022-12-20')
    elif run_num == 2:
        date_series = pd.date_range(start='2023-01-30', end='2023-04-24')

    ids_series = subjects

    # Create a dataframe using a cartesian product of the two series
    df_complete_idDate = pd.DataFrame({
        'ParticipantIdentifier': np.repeat(ids_series, len(date_series)),
        'trial_date': date_series.tolist() * len(ids_series)
    }).reset_index(drop=True)
    # Convert to datetime.date
    df_complete_idDate['trial_date'] = pd.to_datetime(df_complete_idDate['trial_date']).dt.date
    
    return df_complete_idDate


