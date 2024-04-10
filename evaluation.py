import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def evaluate_model(model, X_test, y_test, plot_roc_pr=True):
    """
    Evaluate the performance of a trained machine learning model.
    
    Args:
    model (sklearn.pipeline.Pipeline): Trained machine learning model pipeline.
    X_test (pd.DataFrame): Features of the test dataset.
    y_test (pd.Series): Target variable of the test dataset.
    plot_roc_pr (bool): Whether to plot ROC curve and Precision-Recall curve. Default is True.
    """
    if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
        raise ValueError("Model does not have required methods: predict and predict_proba")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Accuracy, Precision, Recall, F1-score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC AUC:", roc_auc)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    if plot_roc_pr:
        # Plot ROC curve
        plot_roc_curve(model, X_test, y_test)
        plt.title('ROC Curve')
        plt.show()
        
        # Plot Precision-Recall curve
        plot_precision_recall_curve(model, X_test, y_test)
        plt.title('Precision-Recall Curve')
        plt.show()

def interpret_model(model, X, feature_names, plot_type='shap', figsize=(10, 6)):
    """
    Interpret the trained machine learning model.
    
    Args:
    model (sklearn.pipeline.Pipeline): Trained machine learning model pipeline.
    X (pd.DataFrame): Features of the dataset.
    feature_names (list): List of feature names.
    plot_type (str): Type of plot to generate. Options: 'shap' (default) for SHAP summary plot.
    figsize (tuple): Figure size for the plot. Default is (10, 6).
    """
    # Feature Importances
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        print("Feature Importance:\n", importance_df)
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.show()
    else:
        print("Model does not support feature importances.")
    
    # SHAP values (for tree-based models)
    if plot_type == 'shap':
        if hasattr(model, 'predict_proba') and isinstance(model, (shap.TreeExplainer, shap.Explainer)):
            shap_explainer = shap.Explainer(model)
            shap_values = shap_explainer.shap_values(X)
            shap.summary_plot(shap_values, X, feature_names=feature_names)
        else:
            print("Model does not support SHAP values.")
    else:
        print("Invalid plot_type. Choose from 'shap'.")

def main():
    # Example usage:
    X_test, y_test = None, None  # Load your test data
    model = None  # Load your trained model
    feature_names = None  # Provide a list of feature names
    
    evaluate_model(model, X_test, y_test)
    interpret_model(model, X_test, feature_names)

if __name__ == "__main__":
    main()
