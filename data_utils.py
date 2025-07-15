import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath, index=False):
    """
        Load in any filepath, returns the dataframe
        index=False is used when using raw data
        index=True is used when using your own (scaled, info, clean)
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
    """
        Renames any columns, drops duplicates,
        removes NaN, and saves to file
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
    """
        Scales the data and adds "_T" to any columns put through it
    """
    df = load_data(filepath, index)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_values, columns=[col + "_T" for col in df])

    if save_path:
        df_scaled.to_csv(save_path, index=True)
        print(f"Scaled data saved to: {save_path}")
    
    return df_scaled
