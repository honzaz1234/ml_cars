import ast
import pandas as pd
from unidecode import unidecode

def create_share_df(
    df, col, color="blue", min_threshold=None, color_other=None,
    list_colours=None):
    """creates data frame that can be used for plotting shares of observations of individual values of some column
    """

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
    """filters out rows from data frame, in which there are outliers for some  
    specified column
    """

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1
    # Define lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df_filtered

def remove_accents(string):
    return unidecode(string)

def rename_cols(value, dict): 
    """return value based on the dictionary mapping"""
    
    if value in dict:
        return dict[value]
    elif value is None:
        return None
    else:
        return "NC"
    
def get_emissions_standard(row):
    """function for assigning emission standard based on the date of registration of vehicle
    """

    if row >= "2021-01-01":
        return "Euro 6d"
    elif row >= "2019-09-01":
        return "Euro 6d-TEMP"
    elif row >= "2018-09-01":
        return "Euro 6c"
    elif row >= "2015-09-01":
        return "Euro 6b"
    elif row >= "2013-01-01":
        return "Euro 5b"
    elif row >= "2011-01-01":
        return "Euro 5a"
    elif row >= "2006-01-01":
        return "Euro 4"
    elif row >= "2001-01-01":
        return "Euro 3"
    elif row >= "1997-01-01":
        return "Euro 2"
    elif row >= "1993-01-01":
        return "Euro 1"

def get_share(df, col, drop_na=False):
    """function for getting shares of values of some column"""

    if drop_na == False:
        count = df.groupby(col, as_index=False, dropna=False)[col].size()
    else:
        count = df.groupby(col, as_index=False)[col].size()
    count = pd.DataFrame(count)
    count.columns = [col,"n"]
    count["share"] = count["n"]/count["n"].sum()
    return count
    

def unify_val(df, col_name, cut_off=0.9):
        """function for reassigning values of some column based on values of observations that are the same model of car
        cut_off attribute decides how much homogenous the values of the model in the data frame have to be so the value is reassigned;
        example: cut_off=0.9 - 90% of the observations of the same model have to have the same value of the col_name, so the col_name is changed 
        """

        log_df = pd.DataFrame({"model":pd.Series(), "brand":pd.Series(), "type": pd.Series(), "share": pd.Series(), "n_obs": pd.Series(), "n_value": pd.Series()})
        distinct = df.groupby(["brand", "model"], as_index=False)[["brand", "model"]].size()
        for ind in range(0, distinct.shape[0]):
            model_val = distinct.iloc[ind, 1]
            brand_val = distinct.iloc[ind, 0]
            df_val = df[(df["model"] == model_val) & (df["brand"] == brand_val)]
            n_obs = df_val.shape[0]
            shares = get_share(df_val, col_name)
            shares_wNA = get_share(df_val, col_name, drop_na=True)
            n_distinct = shares.shape[0]
            if shares_wNA.shape[0] > 0:
                share_max_value = shares_wNA[shares_wNA["share"] == shares_wNA["share"].max()].iloc[0, 0]
                share_max = shares_wNA["share"].max()
            else:
                share_max_value = None
                share_max = shares["share"].max()
            if shares.shape[0] == 1:
                continue
            if shares_wNA["share"].max() < cut_off:
                log = pd.DataFrame({"brand": brand_val, "model": model_val, "type": "undecided", "share": share_max, "n_obs": n_obs, "n_value": share_max_value, "n_distinct": n_distinct}, index=[0])
                log_df = pd.concat([log_df, log])
                continue
            else:
                df.loc[(df["brand"] == brand_val) & (df["model"] == model_val), col_name] = share_max_value
                log = pd.DataFrame({"brand": brand_val, "model": model_val, "type": "most_other", "share": share_max, "n_obs": n_obs, "n_value": share_max_value, "n_distinct": n_distinct}, index=[0])
                log_df = pd.concat([log_df, log])
        print(log_df.groupby("type").count()) 
        return log_df

def assign_dummy(row, col):
    """assign 1 to column if its column name is found in a list in the col"""

    for val in ast.literal_eval(row[col]):
        row.loc[val] = 1