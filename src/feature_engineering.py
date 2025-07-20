import pandas as pd
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """A class for feature engineering operations."""
    
    @staticmethod
    def create_pokemon_type_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Pokemon type columns to one-hot encoded features.
        
        This approach creates binary features for each Pokemon type, allowing the model
        to capture dual-type Pokemon effectively. This is better than label encoding
        because it doesn't impose an artificial order on the types.
        
        Args:
            df: DataFrame with Type 1 and Type 2 columns
            
        Returns:
            DataFrame with one-hot encoded type features
        """
        logger.info("Creating Pokemon type features")
        
        df_processed = df.copy()
        
        # Fill missing Type 2 values
        df_processed['Type 2'].fillna('SingleType', inplace=True)
        
        # Get all unique types
        unique_types = list(set(df_processed['Type 1'].unique()) | 
                           set(df_processed['Type 2'].unique()))
        unique_types = [t for t in unique_types if t != 'SingleType']
        
        # Create binary features for each type
        for pokemon_type in unique_types:
            df_processed[pokemon_type] = (
                (df_processed['Type 1'] == pokemon_type) | 
                (df_processed['Type 2'] == pokemon_type)
            ).astype(int)
        
        # Drop original type columns
        df_processed.drop(['Type 1', 'Type 2'], axis=1, inplace=True)
        
        logger.info(f"Created {len(unique_types)} type features")
        return df_processed
    
    @staticmethod
    def prepare_pokemon_data(df: pd.DataFrame, target_col: str = 'Legendary', 
                           test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare Pokemon dataset for machine learning.
        
        Args:
            df: Raw Pokemon DataFrame
            target_col: Name of the target column
            test_size: Proportion of test set
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing Pokemon data for machine learning")
        
        # Remove index column if exists
        if '#' in df.columns:
            df = df.drop('#', axis=1)
        
        # Feature engineering
        df_processed = FeatureEngineer.create_pokemon_type_features(df)
        
        # Prepare features and target
        if target_col in df_processed.columns:
            # Move target to the end
            target_series = df_processed.pop(target_col)
            df_processed[target_col] = target_series
            
            # Features: everything except Name and target
            feature_cols = [col for col in df_processed.columns if col not in ['Name', target_col]]
            X = df_processed[feature_cols]
            y = df_processed[target_col]
        else:
            logger.error(f"{target_col} column not found in dataset")
            raise ValueError(f"{target_col} column not found in dataset")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional statistical features from base stats.
        
        Args:
            df: DataFrame with Pokemon stats
            
        Returns:
            DataFrame with additional statistical features
        """
        logger.info("Creating statistical features")
        
        df_enhanced = df.copy()
        
        # Assuming these are the standard Pokemon stat columns
        stat_columns = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        available_stats = [col for col in stat_columns if col in df.columns]
        
        if len(available_stats) >= 2:
            # Total stats
            df_enhanced['Total_Stats'] = df_enhanced[available_stats].sum(axis=1)
            
            # Average stats
            df_enhanced['Avg_Stats'] = df_enhanced[available_stats].mean(axis=1)
            
            # Attack/Defense ratio
            if 'Attack' in df.columns and 'Defense' in df.columns:
                df_enhanced['Attack_Defense_Ratio'] = df_enhanced['Attack'] / (df_enhanced['Defense'] + 1)
            
            # Special Attack/Special Defense ratio
            if 'Sp. Atk' in df.columns and 'Sp. Def' in df.columns:
                df_enhanced['SpAtk_SpDef_Ratio'] = df_enhanced['Sp. Atk'] / (df_enhanced['Sp. Def'] + 1)
            
            # Offensive vs Defensive capabilities
            if all(col in df.columns for col in ['Attack', 'Sp. Atk', 'Defense', 'Sp. Def']):
                df_enhanced['Offensive_Power'] = df_enhanced['Attack'] + df_enhanced['Sp. Atk']
                df_enhanced['Defensive_Power'] = df_enhanced['Defense'] + df_enhanced['Sp. Def']
                df_enhanced['Off_Def_Ratio'] = df_enhanced['Offensive_Power'] / (df_enhanced['Defensive_Power'] + 1)
            
            logger.info(f"Created {len(df_enhanced.columns) - len(df.columns)} statistical features")
        
        return df_enhanced