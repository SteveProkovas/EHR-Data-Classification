import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.feature_selection import SelectKBest, f_classif

def engineer_features(data):
    """
    Perform feature engineering on the EHR dataset.
    
    Args:
    data (pd.DataFrame): DataFrame containing the preprocessed EHR dataset.
    
    Returns:
    pd.DataFrame: DataFrame with engineered features.
    """
    # Example: Engineer new features based on domain knowledge or interactions between existing features
    data['age_squared'] = data['age'] ** 2
    
    # Example: Perform polynomial feature transformation
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    poly_data = poly_features.fit_transform(data[['numerical_feature1', 'numerical_feature2']])
    poly_columns = poly_features.get_feature_names(['numerical_feature1', 'numerical_feature2'])
    data = pd.concat([data, pd.DataFrame(poly_data, columns=poly_columns)], axis=1)
    
    # Example: Discretize numerical features into bins
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    binned_features = discretizer.fit_transform(data[['numerical_feature3']])
    data['numerical_feature3_binned'] = binned_features
    
    # Example: Select K best features using ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=5)
    selected_features = selector.fit_transform(data.drop(['target_variable'], axis=1), data['target_variable'])
    selected_columns = data.columns[selector.get_support(indices=True)].tolist()
    data = data[selected_columns + ['target_variable']]
    
    return data

def main():
    # Load preprocessed data
    data = pd.read_csv('data/processed/cleaned_data.csv')
    
    # Perform feature engineering
    engineered_data = engineer_features(data)
    
    # Save engineered data
    engineered_data.to_csv('data/processed/engineered_data.csv', index=False)
    print("Feature engineering completed. Engineered data saved.")

if __name__ == "__main__":
    main()
