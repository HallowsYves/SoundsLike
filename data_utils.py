import pandas as pd
from sklearn.preprocessing import StandardScaler

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
