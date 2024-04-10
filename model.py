from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, plot_confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def build_model(model_type='ensemble', preprocessing_pipeline=None):
    """
    Build a machine learning model pipeline.
    
    Args:
    model_type (str): Type of machine learning model to build. Options: 'ensemble', 'random_forest', 'logistic_regression', 'svm'.
    preprocessing_pipeline (sklearn.pipeline.Pipeline): Preprocessing pipeline.
    
    Returns:
    sklearn.pipeline.Pipeline: Machine learning model pipeline.
    """
    if model_type == 'ensemble':
        model_rf = RandomForestClassifier(random_state=42)
        model_lr = LogisticRegression(max_iter=1000, random_state=42)
        model_svm = SVC(probability=True, random_state=42)
        model = VotingClassifier(estimators=[('rf', model_rf), ('lr', model_lr), ('svm', model_svm)], voting='soft')
        params = {}  # No hyperparameters to tune for ensemble methods
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
        params = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2']
        }
    elif model_type == 'svm':
        model = SVC(probability=True, random_state=42)
        params = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf']
        }
    else:
        raise ValueError("Invalid model_type. Choose from 'ensemble', 'random_forest', 'logistic_regression', or 'svm'.")
    
    steps = []
    if preprocessing_pipeline:
        steps.append(('preprocessing', preprocessing_pipeline))
    steps.append(('model', model))
    
    pipeline = Pipeline(steps)
    
    return pipeline, params

def train_model(pipeline, params, X_train, y_train):
    """
    Train a machine learning model using cross-validation and hyperparameter tuning.
    
    Args:
    pipeline (sklearn.pipeline.Pipeline): Machine learning model pipeline.
    params (dict): Hyperparameter grid for GridSearchCV.
    X_train (pd.DataFrame): Features of the training dataset.
    y_train (pd.Series): Target variable of the training dataset.
    
    Returns:
    sklearn.pipeline.Pipeline: Trained machine learning model pipeline.
    """
    grid_search = GridSearchCV(pipeline, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of a trained machine learning model.
    
    Args:
    model (sklearn.pipeline.Pipeline): Trained machine learning model pipeline.
    X_test (pd.DataFrame): Features of the test dataset.
    y_test (pd.Series): Target variable of the test dataset.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("ROC AUC:", roc_auc)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(model, X_test, y_test, cmap='Blues', normalize='true')
    plt.title('Confusion Matrix')
    plt.show()

def save_model(model, filepath):
    """
    Save a trained machine learning model to disk.
    
    Args:
    model (sklearn.pipeline.Pipeline): Trained machine learning model pipeline.
    filepath (str): File path to save the trained model.
    """
    joblib.dump(model, filepath)
    print("Model saved successfully.")

def load_model(filepath):
    """
    Load a trained machine learning model from disk.
    
    Args:
    filepath (str): File path to load the trained model.
    
    Returns:
    sklearn.pipeline.Pipeline: Trained machine learning model pipeline.
    """
    model = joblib.load(filepath)
    return model

def main():
    # Example usage:
    X_train, y_train = None, None  # Load your training data
    X_test, y_test = None, None    # Load your test data
    
    preprocessing_pipeline = None  # Add your preprocessing pipeline
    
    # Ensemble Model
    pipeline_ensemble, params_ensemble = build_model(model_type='ensemble', preprocessing_pipeline=preprocessing_pipeline)
    trained_model_ensemble = train_model(pipeline_ensemble, params_ensemble, X_train, y_train)
    print("Ensemble Model:")
    evaluate_model(trained_model_ensemble, X_test, y_test)
    save_model(trained_model_ensemble, 'trained_model_ensemble.pkl')

    # Random Forest
    pipeline_rf, params_rf = build_model(model_type='random_forest', preprocessing_pipeline=preprocessing_pipeline)
    trained_model_rf = train_model(pipeline_rf, params_rf, X_train, y_train)
    print("\nRandom Forest Model:")
    evaluate_model(trained_model_rf, X_test, y_test)
    save_model(trained_model_rf, 'trained_model_rf.pkl')
    
    # Logistic Regression
    pipeline_lr, params_lr = build_model(model_type='logistic_regression', preprocessing_pipeline=preprocessing_pipeline)
    trained_model_lr = train_model(pipeline_lr, params_lr, X_train, y_train)
    print("\nLogistic Regression Model:")
    evaluate_model(trained_model_lr, X_test, y_test)
    save_model(trained_model_lr, 'trained_model_lr.pkl')
    
    # SVM
    pipeline_svm, params_svm = build_model(model_type='svm', preprocessing_pipeline=preprocessing_pipeline)
    trained_model_svm = train_model(pipeline_svm, params_svm, X_train, y_train)
    print("\nSVM Model:")
    evaluate_model(trained_model_svm, X_test, y_test)
    save_model(trained_model_svm, 'trained_model_svm.pkl')
    
    # Load the saved models
    loaded_model_ensemble = load_model('trained_model_ensemble.pkl')
    loaded_model_rf = load_model('trained_model_rf.pkl')
    loaded_model_lr = load_model('trained_model_lr.pkl')
    loaded_model_svm = load_model('trained_model_svm.pkl')
    print("\nModels loaded successfully.")

if __name__ == "__main__":
    main()
