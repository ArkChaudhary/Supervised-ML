import pandas as pd
import numpy as np
import warnings
import logging
import os
from pathlib import Path

# Suppress warnings
warnings.simplefilter(action="ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our custom modules
from eda import DataExplorer
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer


def load_pokemon_data(filepath: str) -> pd.DataFrame:
    """
    Load Pokemon dataset from CSV file.
    
    Args:
        filepath: Path to the Pokemon CSV file
        
    Returns:
        DataFrame with Pokemon data
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data from {filepath}")
        logger.info(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def run_pokemon_analysis(data_path: str = "../data/Pokemon.csv", 
                        target_column: str = "Legendary",
                        test_size: float = 0.2,
                        random_state: int = 42):
    """
    Run complete Pokemon legendary classification analysis.
    
    Args:
        data_path: Path to Pokemon dataset
        target_column: Name of target variable
        test_size: Proportion of test set
        random_state: Random state for reproducibility
    """
    logger.info("Starting Pokemon Legendary Classification Analysis")
    
    try:
        # 1. Load data
        df_raw = load_pokemon_data(data_path)
        
        # 2. Feature Engineering and Data Preparation
        engineer = FeatureEngineer()
        
        # Add statistical features if desired
        df_enhanced = engineer.create_statistical_features(df_raw)
        
        # Prepare data for modeling
        X_train, X_test, y_train, y_test = engineer.prepare_pokemon_data(
            df_enhanced, target_col=target_column, 
            test_size=test_size, random_state=random_state
        )
        
        # 3. Exploratory Data Analysis
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Combine data for EDA
        df_full = pd.concat([X_train, X_test], ignore_index=True)
        df_full[target_column] = pd.concat([y_train, y_test], ignore_index=True)
        
        explorer = DataExplorer()
        explorer.explore_dataframe(df_full, target=target_column)
        
        # 4. Model Training and Evaluation
        print("\n" + "="*80)
        print("MODEL TRAINING AND EVALUATION")
        print("="*80)
        
        trainer = ModelTrainer()
        
        # Optimize KNN hyperparameters
        print("\nOptimizing KNN hyperparameters...")
        best_knn_params = trainer.optimize_knn(X_train, y_train)
        print(f"Best KNN parameters: {best_knn_params}")
        
        # Train and evaluate all models
        results_df = trainer.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Display results
        print("\n" + "="*80)
        print("MODEL PERFORMANCE RESULTS")
        print("="*80)
        print(results_df.round(4).to_string(index=False))
        
        # Visualize results
        trainer.visualize_results(results_df)
        
        # 5. Detailed Analysis of Best Models
        print("\n" + "="*80)
        print("DETAILED MODEL ANALYSIS")
        print("="*80)
        
        # Find best model based on F1 score
        best_model_name = results_df.loc[results_df['F1_Score'].idxmax(), 'Model']
        print(f"\nBest performing model: {best_model_name}")
        
        # Generate classification report for best model
        trainer.generate_classification_report(X_test, y_test, best_model_name)
        
        # Show feature importance for tree-based models
        if 'Tree' in best_model_name or 'Forest' in best_model_name:
            feature_importance = trainer.get_feature_importance(best_model_name, X_train.columns.tolist())
            if not feature_importance.empty:
                print(f"\nTop 10 Most Important Features ({best_model_name}):")
                print(feature_importance.head(10).to_string(index=False))
        
        logger.info("Analysis completed successfully")
        
        return {
            'results': results_df,
            'trainer': trainer,
            'best_model': best_model_name,
            'data': (X_train, X_test, y_train, y_test)
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


def main():
    """Main function to run the analysis."""
    # Set up paths
    current_dir = Path(__file__).parent
    data_path = current_dir / "../data/Pokemon.csv"
    
    # Check if data file exists
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        logger.info("Please ensure Pokemon.csv is in the data/ directory")
        return
    
    # Run analysis
    try:
        results = run_pokemon_analysis(str(data_path))
        logger.info("Analysis completed successfully!")
        
        # Optional: Save results
        output_dir = current_dir / "../outputs"
        output_dir.mkdir(exist_ok=True)
        
        # Save results to CSV
        results_path = output_dir / "model_performance_results.csv"
        results['results'].to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Failed to run analysis: {str(e)}")
        return


if __name__ == "__main__":
    main()