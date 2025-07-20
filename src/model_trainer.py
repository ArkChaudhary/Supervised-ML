import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Tuple

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class ModelTrainer:
    """A class for training and evaluating multiple machine learning models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def get_model_configs(self) -> Dict[str, Any]:
        """Define model configurations."""
        return {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree (Gini)': DecisionTreeClassifier(criterion='gini', random_state=42),
            'Decision Tree (Entropy)': DecisionTreeClassifier(criterion='entropy', random_state=42),
            'K-NN': KNeighborsClassifier(n_neighbors=3),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Naive Bayes': MultinomialNB(),
            'SVC (Linear)': SVC(kernel="linear", random_state=42),
            'SVC (RBF)': SVC(kernel="rbf", random_state=42)
        }
    
    def train_and_evaluate_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
        """
        Train multiple models and evaluate their performance.
        
        Args:
            X_train, X_test: Training and testing features
            y_train, y_test: Training and testing targets
            
        Returns:
            DataFrame with model performance metrics
        """
        logger.info("Starting model training and evaluation")
        
        model_configs = self.get_model_configs()
        results = []
        
        # Store training data for cross-validation
        self.X_train = X_train
        self.y_train = y_train
        
        for model_name, model in model_configs.items():
            logger.info(f"Training {model_name}")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, predictions, model, X_train, y_train)
                metrics['Model'] = model_name
                results.append(metrics)
                
                # Store model and predictions
                self.models[model_name] = model
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results.append({
                    'Model': model_name,
                    'Accuracy': None,
                    'Precision': None,
                    'Recall': None,
                    'F1_Score': None,
                    'Cross_Val_Score': None
                })
        
        results_df = pd.DataFrame(results)
        results_df = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'Cross_Val_Score']]
        
        logger.info("Model training completed")
        return results_df
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive metrics for model evaluation."""
        metrics = {}
        
        try:
            metrics['Accuracy'] = accuracy_score(y_true, y_pred)
            metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['F1_Score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            metrics['Cross_Val_Score'] = cv_scores.mean()
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            metrics = {key: None for key in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Cross_Val_Score']}
        
        return metrics
    
    def optimize_knn(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Optimize KNN hyperparameters using GridSearchCV."""
        logger.info("Optimizing KNN hyperparameters")
        
        param_grid = {
            'n_neighbors': [1, 3, 5, 7, 9],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 40, 50]
        }
        
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(
            estimator=knn, 
            param_grid=param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best KNN parameters: {grid_search.best_params_}")
        return grid_search.best_params_
    
    def visualize_results(self, results_df: pd.DataFrame) -> None:
        """Create visualizations for model comparison."""
        logger.info("Creating result visualizations")
        
        # Remove rows with all None values
        clean_results = results_df.dropna(how='all', subset=['Accuracy', 'Precision', 'Recall', 'F1_Score'])
        
        if clean_results.empty:
            logger.warning("No valid results to visualize")
            return
        
        # Model performance comparison
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in clean_results.columns:
                metric_data = clean_results[['Model', metric]].dropna()
                if not metric_data.empty:
                    sns.barplot(data=metric_data, x='Model', y=metric, ax=axes[i])
                    axes[i].set_title(f'Model Comparison: {metric}')
                    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
                    axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        # Print top performers
        print("\n" + "="*50)
        print("TOP PERFORMING MODELS")
        print("="*50)
        
        for metric in ['Accuracy', 'F1_Score']:
            if metric in clean_results.columns:
                top_model_idx = clean_results[metric].idxmax()
                top_model = clean_results.loc[top_model_idx]
                print(f"Best {metric}: {top_model['Model']} ({top_model[metric]:.4f})")
    
    def generate_classification_report(self, X_test: pd.DataFrame, y_test: pd.Series, 
                                     model_name: str = None) -> None:
        """Generate detailed classification report for a specific model."""
        if model_name is None:
            # Use the best performing model
            if not self.models:
                logger.error("No trained models available")
                return
            model_name = list(self.models.keys())[0]
        
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION REPORT: {model_name}")
        print("="*60)
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def get_feature_importance(self, model_name: str, feature_names: list) -> pd.DataFrame:
        """Get feature importance for tree-based models."""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return pd.DataFrame()
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(15), x='importance', y='feature')
            plt.title(f'Top 15 Feature Importance: {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            
            return importance_df
        else:
            logger.warning(f"Model {model_name} does not have feature_importances_ attribute")
            return pd.DataFrame()