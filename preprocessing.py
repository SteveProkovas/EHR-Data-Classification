import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """
    Load the EHR dataset from the given file path.
    
    Args:
    file_path (str): Path to the EHR dataset CSV file.
    
    Returns:
    pd.DataFrame: Loaded DataFrame containing the dataset.
    """
    try:
        # Load dataset
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while loading the dataset: {str(e)}")
        return None

def preprocess_data(data):
    """
    Preprocess the EHR dataset.
    
    Args:
    data (pd.DataFrame): DataFrame containing the raw EHR dataset.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    if data is None:
        return None
    
    # Remove duplicates or irrelevant columns
    data = remove_irrelevant_columns(data)
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # Handle outliers
    data = handle_outliers(data)
    
    # Encode categorical variables and normalize numerical features
    data = encode_and_normalize_features(data)
    
    return data

def remove_irrelevant_columns(data):
    """
    Remove duplicates or irrelevant columns from the dataset.
    
    Args:
    data (pd.DataFrame): DataFrame containing the raw EHR dataset.
    
    Returns:
    pd.DataFrame: DataFrame with irrelevant columns removed.
    """
    # Example: Remove duplicates
    data.drop_duplicates(inplace=True)
    
    # Example: Remove irrelevant columns based on data analysis
    # data.drop(['Irrelevant_Column1', 'Irrelevant_Column2'], axis=1, inplace=True)
    
    return data

def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    
    Args:
    data (pd.DataFrame): DataFrame containing the raw EHR dataset.
    
    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    # Example: Impute missing values
    imputer = SimpleImputer(strategy='mean')
    data[['numerical_feature1', 'numerical_feature2']] = imputer.fit_transform(data[['numerical_feature1', 'numerical_feature2']])
    
    return data

def handle_outliers(data):
    """
    Handle outliers in numerical features of the dataset.
    
    Args:
    data (pd.DataFrame): DataFrame containing the raw EHR dataset.
    
    Returns:
    pd.DataFrame: DataFrame with outliers handled.
    """
    # Example: Implement outlier detection and handling techniques
    # (e.g., Winsorization, clipping, or removing extreme values)
    # data['numerical_feature1'] = handle_outliers(data['numerical_feature1'])
    
    return data

def encode_and_normalize_features(data):
    """
    Encode categorical variables and normalize numerical features.
    
    Args:
    data (pd.DataFrame): DataFrame containing the raw EHR dataset.
    
    Returns:
    pd.DataFrame: DataFrame with encoded and normalized features.
    """
    # Example: Encode categorical variables and normalize numerical features using ColumnTransformer and Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['categorical_feature1', 'categorical_feature2']),
            ('num', StandardScaler(), ['numerical_feature1', 'numerical_feature2'])
        ], remainder='passthrough')  # Remainder columns are passed through without any transformation
    
    data = pd.DataFrame(preprocessor.fit_transform(data))
    
    return data

def save_data(data, file_path):
    """
    Save preprocessed data to a CSV file.
    
    Args:
    data (pd.DataFrame): Preprocessed DataFrame.
    file_path (str): Path to save the preprocessed data CSV file.
    """
    try:
        data.to_csv(file_path, index=False)
        print(f"Preprocessed data saved to {file_path}")
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the preprocessed data: {str(e)}")

def main():
    # Load dataset
    file_path = 'data/raw/ehr_data.csv'
    data = load_data(file_path)
    
    # Preprocess dataset
    preprocessed_data = preprocess_data(data)
    
    if preprocessed_data is not None:
        # Save preprocessed data
        save_data(preprocessed_data, 'data/processed/cleaned_data.csv')
    else:
        print("Preprocessing failed. Unable to save preprocessed data.")

if __name__ == "__main__":
    main()
