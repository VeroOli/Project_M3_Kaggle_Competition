import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


def remove_outliers(df, deviations=3):

    df_clean = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_mean = df[col].mean()
            col_std = df[col].std()
            min_val = col_mean - deviations * col_std
            max_val = col_mean + deviations * col_std
            df_clean = df_clean[(df_clean[col] >= min_val) & (df_clean[col] <= max_val)]
    return df_clean


def cat_var(df, cols, pd):
    cat_list = []
    for col in cols:
        cat = df[col].unique()
        cat_num = len(cat)
        cat_dict = {"categorical_variable":col,
                    "number_of_possible_values":cat_num,
                    "values":cat}
        cat_list.append(cat_dict)
    df = pd.DataFrame(cat_list).sort_values(by="number_of_possible_values", ascending=False)
    return df.reset_index(drop=True)



def delete_features_corr(df):
    corr_matrix = round(df.corr(numeric_only=True).abs(),2)
    to_drop = corr_matrix.columns[corr_matrix['price'] <= 0.1]
    df_correct = df
    df_correct.drop(to_drop, axis=1, inplace=True)
    return df_correct,to_drop




def remove_outliers(df):
    if 'x' in df.columns:
        df = df[df['x'] < 10]
    if 'y' in df.columns:
        df = df[df['y'] < 10]
    if 'z' in df.columns:
        df = df[(df['z'] < 6.5) & (df['z'] > 2)]
    if 'table' in df.columns:
        df = df[(df['table'] < 70) & (df['table'] > 50)]
    if 'depth' in df.columns:
        df = df[(df['depth'] < 70) & (df['depth'] > 50)]
    return df


def remove_outliers_price(df):
    if 'price' in df.columns:
        df = df[df['price'] < 12500]
    return df


def drop_zeros(df):
    df = df.drop(df[df['x'] == 0].index)
    df = df.drop(df[df['y'] == 0].index)
    df = df.drop(df[df['z'] == 0].index)
    return df


def remove_duplicates(df):
    df = df.drop_duplicates()
    return df


def imputation(df):
    median_x = df.loc[df['x'] != 0, 'x'].median()
    median_y = df.loc[df['y'] != 0, 'y'].median()
    median_z = df.loc[df['z'] != 0, 'z'].median()
    df['x'] = df['x'].replace(0, median_x)
    df['y'] = df['y'].replace(0, median_y)
    df['z'] = df['z'].replace(0, median_z)
    return df



def label_encoder(df):
    df_enc = df.copy()
    for column in df.columns:
        if df[column].dtype == 'object':
            enc_label = LabelEncoder()
            df_enc[column] = enc_label.fit_transform(df[column])
    return df_enc

def onehot_encoder(df):
    df_enc = df.copy()
    categorical_columns = [column for column in df.columns if df[column].dtype == 'object']
    
    df_enc = pd.concat([df_enc.drop(categorical_columns, axis=1),
                        pd.get_dummies(df_enc[categorical_columns], drop_first=True)], axis=1)
    return df_enc

def stardard_scaling_train(df):
    X = df.drop(columns=['price'])
    y = df['price']
    columns = X.columns
    
    numeric_columns = X.select_dtypes(exclude=['object']).columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numeric_columns])
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_columns)
    for col in df.select_dtypes(include=['object']).columns:
        X_scaled_df[col] = df[col]
    X_scaled_df['price'] = y
    
    return X_scaled_df

def robust_scaling_train(df):
    # Separar las características (X) y la variable objetivo (y)
    X = df.drop(columns=['price'])
    y = df['price']
    
    # Identificar las columnas numéricas
    numeric_columns = X.select_dtypes(exclude=['object']).columns
    
    # Aplicar el escalado robusto solo a las columnas numéricas
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X[numeric_columns])
    
    # Convertir el resultado escalado a un DataFrame, manteniendo el mismo índice
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_columns, index=X.index)
    
    # Agregar las columnas categóricas de vuelta al DataFrame escalado
    for col in X.select_dtypes(include=['object']).columns:
        X_scaled_df[col] = X[col].values
    
    # Agregar la columna de precio de vuelta al DataFrame escalado
    X_scaled_df['price'] = y.values
    
    # Asegurar que las columnas estén en el mismo orden que el DataFrame original
    X_scaled_df = X_scaled_df[df.columns.drop('price').to_list() + ['price']]
    
    return X_scaled_df



def robust_scaling_test(df):
    # Identificar las columnas numéricas
    numeric_columns = df.select_dtypes(exclude=['object']).columns
    
    # Aplicar el escalado robusto solo a las columnas numéricas
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[numeric_columns])
    
    # Convertir el resultado escalado a un DataFrame, manteniendo el mismo índice
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_columns, index=df.index)
    
    # Agregar las columnas categóricas de vuelta al DataFrame escalado
    for col in df.select_dtypes(include=['object']).columns:
        X_scaled_df[col] = df[col].values
    
    return X_scaled_df

def robust_scaling_test(df):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
    
    return X_scaled_df

def standard_scaling_test(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
    
    return X_scaled_df

def calculate_log(df, column_name):
    df['log_' + column_name] = np.log(df[column_name])
    return df
                   


def inverse_scaler(df):
    return scal.inverse_transform(df.reshape(-1, 1)).flatten()   
                    
                    
def feature_ing(df_features):
    #print('Dataframe features: ',df_features.head())
    # Test the depth calculate
    df_features['depth_mm'] = (df_features['z']*2)/(df_features['x'] + df_features['y'])
    # Obtain the average girdle diameter
    df_features['avg_girdle'] = (df_features['z'])/(df_features['depth_mm'])
    # Obtain table in mm
    df_features['table_mm'] = (df_features['avg_girdle'])*(df_features['table'])/100
    # Obtain table*depth
    df_features['table_depth'] = (df_features['table'])/(df_features['depth'])
    # Obtain x, y, z
    df_features['xyz'] = (df_features['x'])*(df_features['y'])*(df_features['z'])
    return df_features                   
                    
def transformation_data(df, type_data):
    trans_df = classify_shape(df)
    trans_df = encoder(trans_df)
    #trans_df = imputation(trans_df)

    if type_data == 'train_data':
        #trans_df = drop_zeros(trans_df)
        #trans_df = remove_outliers(trans_df)
        #trans_df = remove_duplicates(trans_df)
        selection_features = ['cut', 'color', 'clarity', 'city', 'carat', 'depth', 'shape', 'x', 'y', 'z', 'price']
        #selection_features = ['cut', 'color', 'clarity', 'city', 'carat', 'depth', 'table','x', 'y', 'z', 'price']
    
    if type_data == 'test_data':
        selection_features = ['cut', 'color', 'clarity', 'city', 'carat', 'depth', 'shape', 'x', 'y', 'z']     
        #selection_features = ['cut', 'color', 'clarity', 'city', 'carat', 'depth', 'table', 'x', 'y', 'z']
        
    trans_df = feature_ing(trans_df)
    
    # Calculate others features
    trans_df = calculate_log(trans_df, 'carat')
    trans_df = calculate_log(trans_df, 'x')
    trans_df = calculate_log(trans_df, 'y')
    trans_df = calculate_log(trans_df, 'z')
    trans_df['ratio_length_width'] = trans_df['x']/trans_df['y']
    trans_df['ratio_length_width_depth'] = trans_df['x']/trans_df['y']/trans_df['z']
    trans_df['volume'] = trans_df['x']*trans_df['y']*trans_df['z']
    trans_df['density'] = trans_df['carat']/trans_df['volume']
    
    # Only used selection features
    trans_df_2 = trans_df[selection_features]
    trans_df_2.head()
    
    return trans_df_2, selection_features
                
