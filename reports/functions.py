import pandas as pd

def create_share_df(
    df, col, color="blue", min_threshold=None, color_other=None,
    list_colours=None):
    counts = df[col].value_counts()
    df_counts = pd.DataFrame()
    df_counts["counts"] = counts
    df_counts["share"] = df_counts["counts"] / df.shape[0]
    colour_list = [color for col in df_counts.index]
    if min_threshold is not None:
        df_counts.reset_index(inplace=True)
        other_val = "other_" + col
        df_counts.loc[df_counts["counts"] < min_threshold, col] = other_val
        df_counts = (df_counts.groupby(col)[["counts", "share"]]
                    .sum()
                    .sort_values(ascending=False, by="share"))
        colour_list = [color_other if col ==other_val else color for col in df_counts.index]
    if list_colours is not None:
        colour_list = list_colours
    df_counts["colours"] = colour_list
    return df_counts

def filter_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1
    # Define lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df_filtered