from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def normalize_dataframe(df):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(
        data=scaler.fit_transform(df.values), 
        columns=df.columns, 
        index=df.index)
    return df_normalized

def my_transformation(df, norm=False, drop_nans=True, drop_dupl=False my_decision=0):
    """
    Function to modify original dataframe and return the modified version.
    Change where neccessary.
    """
    # LabelEncoder
    le = LabelEncoder()

    df_modified = df.copy()
    if drop_nans:
        # 0. removes columns of choice
        df_modified.drop(["9", "11"], 1, inplace=True)
    else:
        # Converts NaN values to the mean value of the same column.
        df_modified["9"] = df_modified["9"].fillna(df_modified["9"].mean())
        df_modified["11"] = df_modified["11"].fillna(df_modified["11"].mean())

    if drop_dupl:
        #1 remove duplicated values
        df_modified.drop_duplicates(subset=None, keep='first', inplace=True)

    # 2.Change columns from object to numeric and save them in a variable. 
    X_categorical_no_numbers = df_modified[df_modified.select_dtypes('object').columns].apply(le.fit_transform)
    # 3. Get only numerical columns
    X_others = df_modified.select_dtypes(exclude=['object'])
    if norm:
        # 4. Normalize numerical columns
        X_others = normalize_dataframe(df=X_others)
    
    # 5. concatenate final result
    df_modified = pd.concat([X_others, X_categorical_no_numbers], axis=1)

    if my_decision == 1:
        # To only keep selected columns, to be changed accordingly.  
        # [admission_deposit, city_code_patient, age, hospital_code, available extra rooms in hospital, hospital_type_code_feat_hospital_code, visitors_with_patient, bed grade, severity of illness]
        if drop_nans:
            df_modified = df_modified[["16", "15", "13", "14", "1", "5", "2"]]
        else:
            df_modified = df_modified[["16","11", "15", "13", "14", "1", "5", "2", "9"]]

    return df_modified