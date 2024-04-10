import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

def impute_missing_values(df):
    """
    Impute missing values in the DataFrame.
    
    Args:
    df (pd.DataFrame): Input DataFrame with missing values.
    
    Returns:
    pd.DataFrame: DataFrame with missing values imputed.
    """
    imputer = SimpleImputer(strategy='mean')  # Impute missing numerical values with mean
    df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    return df_filled

def encode_categorical_variables(df):
    """
    Encode categorical variables in the DataFrame using one-hot encoding.
    
    Args:
    df (pd.DataFrame): Input DataFrame with categorical variables.
    
    Returns:
    pd.DataFrame: DataFrame with categorical variables encoded.
    """
    encoder = OneHotEncoder(drop='first', sparse=False)  # One-hot encoding with dropping first category to avoid multicollinearity
    df_encoded = pd.DataFrame(encoder.fit_transform(df), columns=encoder.get_feature_names(df.columns), index=df.index)
    return df_encoded

def normalize_numerical_features(df):
    """
    Normalize numerical features in the DataFrame.
    
    Args:
    df (pd.DataFrame): Input DataFrame with numerical features.
    
    Returns:
    pd.DataFrame: DataFrame with numerical features normalized.
    """
    scaler = StandardScaler()  # Standardize numerical features to have mean=0 and variance=1
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_normalized

def select_features(X, y, k=10):
    """
    Select the top k features using ANOVA F-value between feature and target variable.
    
    Args:
    X (pd.DataFrame): Features DataFrame.
    y (pd.Series): Target variable.
    k (int): Number of features to select.
    
    Returns:
    pd.DataFrame: DataFrame with selected features.
    """
    selector = SelectKBest(score_func=f_classif, k=k)  # Select top k features based on ANOVA F-value
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

def preprocess_data(df, categorical_features, numerical_features, target_variable):
    """
    Preprocess the EHR data including missing value imputation, encoding categorical variables,
    normalization of numerical features, and feature selection.
    
    Args:
    df (pd.DataFrame): Input DataFrame containing EHR data.
    categorical_features (list): List of column names for categorical features.
    numerical_features (list): List of column names for numerical features.
    target_variable (str): Name of the target variable.
    
    Returns:
    X (pd.DataFrame): Preprocessed features DataFrame.
    y (pd.Series): Target variable Series.
    """
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    
    # Define preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Impute missing categorical values
        ('encoder', OneHotEncoder(drop='first', sparse=False))  # One-hot encoding with dropping first category
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing numerical values with mean
        ('scaler', StandardScaler())  # Standardize numerical features
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])
    
    # Apply preprocessing pipeline
    X_preprocessed = pd.DataFrame(preprocessor.fit_transform(X), columns=categorical_features + numerical_features)
    
    # Select top features
    X_selected = select_features(X_preprocessed, y)
    
    return X_selected, y
