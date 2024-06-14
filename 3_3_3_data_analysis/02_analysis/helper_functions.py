import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

# create new cat column based on whether the subject recorded their gap
def convert_column_to_binary(df, column_name):
    """
    Convert a specified column of a DataFrame to binary form where
    1 represents a non-missing value and 0 represents a missing value (NaN).
    
    Args:
    df (pd.DataFrame): The DataFrame containing the column to be converted.
    column_name (str): The name of the column to convert.
    
    Returns:
    pd.Series: A Series with binary values indicating the presence or absence of data.
    """
    binary_series = df[column_name].notna().astype(int)
    boolean_series = binary_series.map({1: True, 0: False})
    return boolean_series

df = pd.read_csv('../02_analysis/df_good.csv')

def subject_df_no_feat_eng(df, sub_num):

    # Convert 'trial_date' to datetime and sort the dataframe
    df['trial_date'] = pd.to_datetime(df['trial_date'])
    # Sort
    df = df.sort_values(by=['PID', 'trial_date'])

    # Subject PID
    subject = np.unique(df.PID)[sub_num]

    # Convert all columns to numeric, setting errors='coerce' will convert non-convertible values to NaN
    df_numeric = df.loc[df.PID == subject].apply(pd.to_numeric, errors='coerce')
    
    X = df_numeric.copy()
    
    # Extract categorical features
    cat_cols = ['day_of_week', 'task_nback_mode']
    X.drop(columns=cat_cols, inplace=True)
    
    # Remove app/web columns for device usage as these are then totaled
    web_cols = [col for col in X.columns if 'passive_sk_device_web_usage' in col]
    app_cols = [col for col in X.columns if 'passive_sk_device_app_usage' in col]

    X.drop(columns=web_cols, inplace=True)
    X.drop(columns=app_cols, inplace=True)
    
    # Remove trial date - using day
    day = X.day
    X.drop(columns=['trial_date', 'day'], inplace=True)
    
    
    # remove existing residual columns since we will be calculating differences/residuals for all columns
    cols = [col for col in X.columns if 'residual' in col]
    X.drop(columns=cols, inplace=True)
    
    # z scored some rts, can remove those
    cols = [col for col in X.columns if 'rt_z' in col]
    X.drop(columns=cols, inplace=True)
    
    # Using the average RT
    X.drop(['task_rt_1', 'task_rt_2', 'task_rt_3', 'task_rt_4'], axis=1, inplace=True)
    
    # remove keyboard sentiment for emoji and word individually since they are combined
    cols = [col for col in X.columns if 'keyboard_sentiment_emoji' in col]
    X.drop(columns=cols, inplace=True)
    
    # remove keyboard sentiment for emoji and word individually since they are combined
    cols = [col for col in X.columns if 'keyboard_sentiment_word' in col]
    X.drop(columns=cols, inplace=True)
    
    # remove device sentiment as this is an error
    cols = [col for col in X.columns if 'device_sentiment' in col]
    X.drop(columns=cols, inplace=True)
    
    # Remove simple gap which is just sr_DAILY_past24_gap
    X.drop(columns='sr_gap_simple', inplace=True)
    
    # Remove columns with NaN counts about threshold
    threshold_nan = 0.5

    nan_counts = X.isna().sum()
    cols_to_drop_due_to_nans = nan_counts[nan_counts > len(X) * threshold_nan].index
    X = X.drop(columns=cols_to_drop_due_to_nans)
    
    # print(f'Removed {len(cols_to_drop_due_to_nans)} features due to NaNs...')
    
    # Scale using Min-Max Normalization
    # print("Applying min-max scaling...")
    X = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))

    # Remove columns with almost no variance
    # Set threshold
    var_threshold = .01

    no_var_cols = X.loc[:, np.var(X) <= var_threshold].columns
    ## X = X.loc[:, np.var(X) > var_threshold]

    # print(f'Removed {len(no_var_cols)} features due to Var <= {var_threshold} after scaling...')
    
    # if the max and min value were the same var they now equal NaN...remove these
    nan_counts = X.isna().sum()
    cols_to_drop_due_to_nans = nan_counts[nan_counts > len(X) * threshold_nan].index
    X = X.drop(columns=cols_to_drop_due_to_nans)

    # print(f'Removed {len(cols_to_drop_due_to_nans)} features due to NaNs after scaling...')
    
    # Compute the correlation matrix - need at least 20 values
    corr_matrix = X.corr(method='pearson', min_periods=20).abs()
    
    # Define the threshold
    threshold = 0.95
    
    # Identify pairs of highly correlated features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Drop highly correlated columns
    X = X.drop(to_drop, axis=1)
    
    return X

def subject_df(df, sub_num):

    # Convert 'trial_date' to datetime and sort the dataframe
    df['trial_date'] = pd.to_datetime(df['trial_date'])
    # Sort
    df = df.sort_values(by=['PID', 'trial_date'])

    # Subject PID
    subject = np.unique(df.PID)[sub_num]

    # Convert all columns to numeric, setting errors='coerce' will convert non-convertible values to NaN
    df_numeric = df.loc[df.PID == subject].apply(pd.to_numeric, errors='coerce')
    
    X = df_numeric.copy()
    
    # Extract categorical features
    cat_cols = ['day_of_week', 'task_nback_mode']
    X.drop(columns=cat_cols, inplace=True)
    
    # Remove app/web columns for device usage as these are then totaled
    web_cols = [col for col in X.columns if 'passive_sk_device_web_usage' in col]
    app_cols = [col for col in X.columns if 'passive_sk_device_app_usage' in col]

    X.drop(columns=web_cols, inplace=True)
    X.drop(columns=app_cols, inplace=True)
    
    # Remove trial date - using day
    day = X.day
    X.drop(columns=['trial_date', 'day'], inplace=True)
    
    
    # remove existing residual columns since we will be calculating differences/residuals for all columns
    cols = [col for col in X.columns if 'residual' in col]
    X.drop(columns=cols, inplace=True)
    
    # z scored some rts, can remove those
    cols = [col for col in X.columns if 'rt_z' in col]
    X.drop(columns=cols, inplace=True)
    
    # Using the average RT
    X.drop(['task_rt_1', 'task_rt_2', 'task_rt_3', 'task_rt_4'], axis=1, inplace=True)
    
    # remove keyboard sentiment for emoji and word individually since they are combined
    cols = [col for col in X.columns if 'keyboard_sentiment_emoji' in col]
    X.drop(columns=cols, inplace=True)
    
    # remove keyboard sentiment for emoji and word individually since they are combined
    cols = [col for col in X.columns if 'keyboard_sentiment_word' in col]
    X.drop(columns=cols, inplace=True)
    
    # remove device sentiment as this is an error
    cols = [col for col in X.columns if 'device_sentiment' in col]
    X.drop(columns=cols, inplace=True)
    
    # Remove simple gap which is just sr_DAILY_past24_gap
    X.drop(columns='sr_gap_simple', inplace=True)
    
    # Remove columns with NaN counts about threshold
    threshold_nan = 0.5

    nan_counts = X.isna().sum()
    cols_to_drop_due_to_nans = nan_counts[nan_counts > len(X) * threshold_nan].index
    X = X.drop(columns=cols_to_drop_due_to_nans)
    
    # print(f'Removed {len(cols_to_drop_due_to_nans)} features due to NaNs...')
    
    # Scale using Min-Max Normalization
    # print("Applying min-max scaling...")
    X = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))

    # Remove columns with almost no variance
    # Set threshold
    var_threshold = .01

    no_var_cols = X.loc[:, np.var(X) <= var_threshold].columns
    ## X = X.loc[:, np.var(X) > var_threshold]

    # print(f'Removed {len(no_var_cols)} features due to Var <= {var_threshold} after scaling...')
    
    # if the max and min value were the same var they now equal NaN...remove these
    nan_counts = X.isna().sum()
    cols_to_drop_due_to_nans = nan_counts[nan_counts > len(X) * threshold_nan].index
    X = X.drop(columns=cols_to_drop_due_to_nans)

    # print(f'Removed {len(cols_to_drop_due_to_nans)} features due to NaNs after scaling...')
    
    # Compute the correlation matrix - need at least 20 values
    corr_matrix = X.corr(method='pearson', min_periods=20).abs()
    
    # Define the threshold
    threshold = 0.95
    
    # Identify pairs of highly correlated features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Drop highly correlated columns
    X = X.drop(to_drop, axis=1)
    
    # print(X.shape)
    
    ################
    # FEATURE ENG. #
    ################
    
    # For each column calculate:

    # - 3 and 7 day rolling average (up to current day)
    # - Expanding average (1st day to current day)
    # - 7 day variance (up to current day)
    # - 1, 2 and 3 day lags of each feature
    # - Difference between 3, 7 and expanding average and current day
    
    # Get numeric columns only for processing
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Dictionary to hold new columns
    new_columns = {}

    # Calculate rolling averages and store in the dictionary
    for col in numeric_cols:
        for window in [3, 7]:
            rolling_col = X[col].rolling(window=window, min_periods=1).mean()  # use min_periods=1 to ensure output
            new_columns[f'{col}_rolling_mean_{window}'] = rolling_col
            new_columns[f'{col}_exponential_rolling_mean_{window}'] = X[col].ewm(span=window).mean()

    # Calculate rolling mean including the current row for each numeric column
    for col in numeric_cols:
        new_columns[f'{col}_rolling_mean_expanding'] = X[col].expanding(min_periods=1).mean()
            
    # Calculate dynamically adjusted EMA
    def dynamically_adjusted_ema(series):
        emas = []
        ema = series.iloc[0]  # initialize EMA with the first data point
        emas.append(ema)
        for t in range(1, len(series)):
            alpha_t = 2 / (t + 1 + 1)  # dynamically adjust alpha
            ema = alpha_t * series.iloc[t] + (1 - alpha_t) * ema
            emas.append(ema)
        return pd.Series(emas, index=series.index)
    for col in numeric_cols:
        new_columns[f'{col}_exponential_rolling_mean_expanding'] = dynamically_adjusted_ema(X[col])
            
    # Calculate rolling variance
    for col in numeric_cols:
        for window in [7]:
            rolling_col = X[col].rolling(window=window, min_periods=1).var()  # use min_periods=1 to ensure output
            new_columns[f'{col}_rolling_var_{window}'] = rolling_col

    # Generate lag features and store in dictionary
    for col in numeric_cols:
        for lag in [1, 2, 3]:
            new_columns[f'{col}_lag_{lag}'] = X[col].shift(lag)

    # Add new_columns to X before calculating differences to ensure all columns exist
    X = pd.concat([X, pd.DataFrame(new_columns)], axis=1)

    # Dictionary to hold new columns
    new_columns = {}

    # Store differences from rolling averages to mean and from current day value to mean
    for col in numeric_cols:
        new_columns[f'{col}_diff_rolling_mean_3_day'] = X[col] - X[f'{col}_rolling_mean_3'] 
        new_columns[f'{col}_diff_rolling_mean_7_day'] = X[col] - X[f'{col}_rolling_mean_7']

    # Store differences from the expanding mean
    for col in numeric_cols:
        new_columns[f'{col}_diff_to_expanding_mean'] = X[col] - X[f'{col}_rolling_mean_expanding']
        
    # Add new_columns to X
    X = pd.concat([X, pd.DataFrame(new_columns)], axis=1)
    
    ############
    # RE-CLEAN #
    ############
    
    # Remove columns with NaN counts about threshold
    threshold_nan = 0.5

    nan_counts = X.isna().sum()
    cols_to_drop_due_to_nans = nan_counts[nan_counts > len(X) * threshold_nan].index
    X = X.drop(columns=cols_to_drop_due_to_nans)

    # print(f'Removed {len(cols_to_drop_due_to_nans)} features due to NaNs...')
    
    # Remove columns with NaN counts about threshold
    threshold_nan = 0.5

    nan_counts = X.isna().sum()
    cols_to_drop_due_to_nans = nan_counts[nan_counts > len(X) * threshold_nan].index
    X = X.drop(columns=cols_to_drop_due_to_nans)

    # print(f'After feature engineering removed {len(cols_to_drop_due_to_nans)} features due to NaNs...')
        
    # Remove columns with almost no variance
    # First scale using Min-Max Normalization
    X = (X-X.min())/(X.max()-X.min())

    # Set threshold
    var_threshold = .01

    no_var_cols = X.loc[:, np.var(X) <= var_threshold].columns
    ## X = X.loc[:, np.var(X) > var_threshold]

    # print(f'Removed {len(no_var_cols)} features due to Var <= {var_threshold}...')
        
    # if the max and min value were the same var = NaN...remove these
    nan_counts = X.isna().sum()
    cols_to_drop_due_to_nans = nan_counts[nan_counts > len(X) * threshold_nan].index
    X = X.drop(columns=cols_to_drop_due_to_nans)

    # print(f'Removed {len(cols_to_drop_due_to_nans)} features due to NaNs...')
    
    # Compute the correlation matrix - need at least 20 values
    corr_matrix = X.corr(method='pearson', min_periods=20).abs()
    
    # Define the threshold
    threshold = 0.95

    # Create a mask to find correlations greater than the threshold, ignoring the diagonal
    mask = abs(corr_matrix) > threshold
    np.fill_diagonal(mask.values, False)

    # Extract pairs above threshold
    high_corr_pairs = corr_matrix[mask].stack()
    
    # Identify pairs of highly correlated features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold) and column != 'sr_gap_heuristic']
    
    # Drop highly correlated columns
    X = X.drop(to_drop, axis=1)
    
    # Dummy Code Categorical features
    X['sr_gap_entry'] = convert_column_to_binary(X, 'sr_gap_heuristic')
    # Defragment
    X = X.copy()
    
    df_sub = df.loc[df.PID == subject]
    cat_df = df_sub[cat_cols]
    
    weekdays = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }

    # Use the map function to convert integers to weekday names
    cat_df['day_of_week'] = cat_df['day_of_week'].map(weekdays)
    
    # Add adherence column
    cat_df = pd.concat([cat_df, X['sr_gap_entry']], axis=1) 

    # Convert to dummy variables without dropping any column
    cat_df_dummies = pd.get_dummies(cat_df, drop_first=True)

    # Explicitly drop the column for adherence
    X.drop('sr_gap_entry', axis=1, inplace=True)
    
    # Add categorical columns to X
    X = pd.concat([X, cat_df_dummies], axis=1)
    
    # Add day back
    X['day'] = day
    
    return X