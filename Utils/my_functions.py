import pandas as pd
import numpy as np


"""functiones to clean strings"""
def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]', '', value)
        value = value.title()
        result.append(value)
    return result



# Data mining:

def normalize_dataframe(df):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(
        data=scaler.fit_transform(df.values), 
        columns=df.columns, 
        index=df.index)
    return df_normalized

# Esta función debería ser usada al principio del todo ya que se realizan los cambios de la misma manera.
def my_transformation(df, norm=False, drop_nans=True, my_decision=0):
    """
    Esta función utiliza un dataframe original, y lo modifica para retornar el dataframe con todos los cambios realizados que hemos creído convenientes.
    """
    df_modified = df.copy()
    if drop_nans:
        # 1. Elimino columna 9 y 11 por tener nan
        df_modified.drop(["9", "11"], 1, inplace=True)
    else:
        # Convierto NaNs a la mean de esas columnas
        df_modified["9"] = df_modified["9"].fillna(df_modified["9"].mean())
        df_modified["11"] = df_modified["11"].fillna(df_modified["11"].mean())

    # 2, 3 y 4. Realizo el encoder de las categóricas además de normalizar
    # 2.Paso las columnas "Object" a numéricas
    X_categorical_no_numbers = df_modified[df_modified.select_dtypes('object').columns].apply(LabelEncoder().fit_transform)
    # 3. Cojo solo columnas numéricas
    X_others = df_modified.select_dtypes(exclude=['object'])
    if norm:
        # 4. Normalizo las columnas numéricas
        X_others = normalize_dataframe(df=X_others)
    # 5. Concateno el resultado final
    df_modified = pd.concat([X_others, X_categorical_no_numbers], axis=1)

    if my_decision == 1:
        # me quedo solo con las columnas: 
        # [admission_deposit, city_code_patient, age, hospital_code, available extra rooms in hospital, hospital_type_code_feat_hospital_code, visitors_with_patient, bed grade, severity of illness]
        df_modified = df_modified[["16","11", "15", "13", "14", "1", "5", "2", "9"]]

    return df_modified

#To save a model
    def save_model(to_save, filepath):
        try:
            if file_exists(filepath=filepath):
                filepath = rename_filename(filepath=filepath)
                pickle.dump(to_save, open(filepath, 'wb'))
                print("Saved successfully")
                return True, filepath
        except Exception as e:
            print("Error during saving model:\n", e)
            return False, filepath