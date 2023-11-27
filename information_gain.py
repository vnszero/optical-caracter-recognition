import pandas as pd
import math
from typing import Union

def entropy(df_data:pd.DataFrame, name_col_category:str) -> float:
    """
        Calculate entropy based on df_data (DataFrame) and category.

        df_data: Data to be used in entropy expression
        name_col_category: name of column (en df_data) that represents category
    """
    # amount_count_col stores, for each category value, its amount

    amount_count_col = df_data[name_col_category].value_counts() 
    num_total = len(df_data)
    entropy = 0

    # Iterate over amount_count_col to find entropy
    for val_feature, count_feature in amount_count_col.items():
        # val_prob must be instance ratio of category
        val_prob = count_feature / num_total
        entropy += math.log(val_prob, 2) * val_prob * -1
    return entropy


def conditional_information_gain(df_data: pd.DataFrame, val_entropy_y:float, name_col_category:str, feature_name:str, feature_val:Union[int,float,str,bool]) ->float:
    """
        Calculates IG(Y|feature_name=feature_val), in other words,
        calculates the information gain of 'feature_name' when it holds value 'feature_val'.
        The entropy(Y) value has been already calculated and it is stored in val_entropy_y.

        df_data: Data to be analysed
        val_entropy_y: entropy(Y)
        name_col_category: name of column which represent a category
        feature_name: feature name to find info gain
        feature_val: value of the feature in this call
                     values can be (boolean, int, float or str)
    """

    df_data_filter = df_data[df_data[feature_name] == feature_val]

    # use df_data_filter to get entropy(Y|feature_name=feature_val) value
    conditional_val_ent = entropy(df_data_filter, name_col_category)

    # use conditional_val_ent to calculate IG(Y|feature_name=feature_val)
    val_ig = val_entropy_y - conditional_val_ent

    return val_ig


def information_gain(df_data:pd.DataFrame, name_col_category:str, feature_name:str) -> float:
    """
        Calculate IG(Y| feature_name), in other words, the info gain for feature_name.

        df_data: Data to be analysed
        name_col_category: name of column which represent a category
        feature_name: feature name to find info gain
        feature_val: value of the feature in this call
    """

    amount_count_col = df_data[feature_name].value_counts()

    val_entropy_y = entropy(df_data, name_col_category)

    num_total = len(df_data)
    val_info_gain = 0
    for val_feature, count_feature in amount_count_col.items():
        val_prob = count_feature / num_total 
        val_info_gain += val_prob * conditional_information_gain(df_data, val_entropy_y, name_col_category, feature_name, val_feature)

    return val_info_gain

def discrete_information_gain(df_data:pd.DataFrame, name_col_category:str, feature_name:str, num_bins: float) -> float:
    """
        Calculate IG(Y| feature_name), in other words, the info gain for feature_name.

        df_data: Data to be analysed
        name_col_category: name of column which represent a category
        feature_name: feature name to find info gain
        feature_val: value of the feature in this call
        num_bins: number of intervals to discrete feature
    """
    df = df_data.copy(deep=True)
    df['discretized_feature'] = pd.cut(df_data[feature_name], bins=num_bins)
    
    val_entropy_y = entropy(df, name_col_category)
    num_total = len(df)
    val_info_gain = 0
    
    for val_bin in df['discretized_feature'].unique():
        val_prob = len(df[df['discretized_feature'] == val_bin]) / num_total
        val_info_gain += val_prob * conditional_information_gain(df, val_entropy_y, name_col_category, 'discretized_feature', val_bin)

    return val_info_gain
