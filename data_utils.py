import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def validate_scaled_data(df, suffix="_T", mean_tol=0.1, std_tol=0.1):
    """Verify that a dataframe contains standardized features.

    The application expects all feature columns to be produced by
    :func:`scale_data`, which standardizes values and appends a ``_T`` suffix
    to each column name.  This helper checks both assumptions and raises a
    ``ValueError`` if any column appears to be unscaled.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        suffix (str, optional): Expected suffix for transformed columns.
        mean_tol (float, optional): Allowed absolute deviation from zero
            for column means.
        std_tol (float, optional): Allowed absolute deviation from one for
            column standard deviations.

    Raises:
        ValueError: If any column is missing the suffix or statistics fall
            outside the tolerated range, indicating unscaled data.
    """

    # Check that all column names include the expected suffix
    missing_suffix = [col for col in df.columns if not col.endswith(suffix)]
    if missing_suffix:
        raise ValueError(
            "Detected columns without the expected suffix '{}': {}".format(
                suffix, missing_suffix
            )
        )

    # Verify statistical properties roughly match standardized data
    means = df.mean()
    stds = df.std()
    bad_stats = [
        col
        for col in df.columns
        if abs(means[col]) > mean_tol or abs(stds[col] - 1) > std_tol
    ]
    if bad_stats:
        raise ValueError(
            "Columns appear unscaled based on mean/std checks: {}".format(
                bad_stats
            )
        )


def validate_scaled_array(arr, mean_tol=0.1, std_tol=0.1):
    """Verify that a numpy array contains standardized features.

    Args:
        arr (np.ndarray): Array to validate where columns represent features.
        mean_tol (float, optional): Allowed absolute deviation from zero for
            feature means.
        std_tol (float, optional): Allowed absolute deviation from one for
            feature standard deviations.

    Raises:
        ValueError: If statistical properties fall outside tolerated ranges,
            indicating the array may be unscaled.
    """

    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    if np.any(np.abs(means) > mean_tol) or np.any(np.abs(stds - 1) > std_tol):
        raise ValueError(
            "Array appears unscaled based on mean/std checks."
        )

def load_data(filepath, index=False):
    """Loads any csv files

    Load in any filepath, returns the dataframe. 
    Aside from error checking if the file is found,
    it provides the user to use the first column in
    their dataframe or not.

    Args:
        filepath: any filepath to a source of data
        index: False for raw data; True when using your own (scaled, data, clean)
    
    Returns:
        A dataframe from pandas
    """
    try:
        if index:
            df = pd.read_csv(filepath, index_col=0)
        else:
            df = pd.read_csv(filepath)
        print(f"found {filepath}")
    except FileNotFoundError:
        print(f"ERROR: could not find {filepath}")
        return None
    
    return df

def clean_data(filepath, index=False, rename=None, duplicates=None, keep=None, save_path=None):
    """Cleans up dataframes and saves them

    Args:
        filepath: uses the load_data() to load into a pandas dataframe.
        index: sets the appropriate index shift in the rows.
        rename: takes in a dictionary with the old and new names.
        duplicates: takes a list with columns to drop duplicates in.
        keep: takes a list of columns to drop rows with NaN values in and make the new df.
        save_path: creates a new csv for the clean dataset.
    
    Returns:
        The cleaned dataframe
    """
    df = load_data(filepath, index)

    if rename:
        df.rename(columns=rename, inplace=True)
    if duplicates:
        df.drop_duplicates(subset=duplicates, inplace=True)
    if keep:
        df.dropna(subset=keep, inplace=True)
        df = df[keep]
    if save_path:
        df.to_csv(save_path, index=True)
        print(f"Clean data saved to: {save_path}")

    df.reset_index(drop=True, inplace=True)
    return df

def scale_data(filepath, index=False, save_path=None):
    """Scales the data and adds "_T" to any columns put through it

    Using StandardScaler, the dataframe is scaled according
    to its values. Then it loops through each column to add a
    "_T" for "Transformed". Finally it saves it to a filepath.

    Args:
        filepath: uses the load_data() to load into a pandas dataframe.
        index: sets the appropriate index shift in the rows.
        save_path: creates a new csv for the scaled dataset.
    
    Returns:
        The scaled dataframe
    """
    df = load_data(filepath, index)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_values, columns=[col + "_T" for col in df])

    if save_path:
        df_scaled.to_csv(save_path, index=True)
        print(f"Scaled data saved to: {save_path}")
    
    return df_scaled
