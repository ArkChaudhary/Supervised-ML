import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional

# Set up plotting style
plt.rc("font", size=14)
sns.set_style("whitegrid")
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class DataExplorer:
    """A class for exploratory data analysis."""
    
    @staticmethod
    def explore_dataframe(df: pd.DataFrame, target: Optional[str] = None, max_cat_cardinality: int = 20) -> None:
        """
        Data Analysis function for any dataframe.
        """
        logger.info("Starting comprehensive data exploration")
        
        print("=" * 80)
        print("DATAFRAME OVERVIEW")
        print("=" * 80)
        print(f"Shape: {df.shape}")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nColumn Data Types:")
        print(df.dtypes.to_string())
        
        # Missing values analysis
        missing_data = df.isnull().sum()
        if missing_data.any():
            print("\nMissing Values:")
            print(missing_data[missing_data > 0].to_string())
            
            # Visualize missing data
            plt.figure(figsize=(10, 6))
            missing_data[missing_data > 0].plot(kind='bar')
            plt.title("Missing Values by Column")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
        # Basic statistics
        print("\nNumerical Statistics:")
        print(df.describe().round(2).to_string())
        
        DataExplorer._analyze_target(df, target)
        DataExplorer._create_correlation_heatmap(df)
        DataExplorer._plot_distributions(df)
        DataExplorer._analyze_categorical_features(df, max_cat_cardinality)
        
        logger.info("Data exploration completed")
    
    @staticmethod
    def _analyze_target(df: pd.DataFrame, target: Optional[str]) -> None:
        """Analyze target variable distribution."""
        if not target or target not in df.columns:
            return
            
        print(f"\n{'='*20} TARGET ANALYSIS {'='*20}")
        print(f"Target Variable: {target}")
        
        target_counts = df[target].value_counts()
        print("\nTarget Distribution:")
        print(target_counts.to_string())
        
        # Create target distribution plot
        plt.figure(figsize=(10, 6))
        if df[target].dtype in ['bool', 'object'] or df[target].nunique() < 10:
            sns.countplot(data=df, x=target)
            plt.title(f"Distribution of {target}")
            
            # Add percentage labels
            total = len(df)
            for p in plt.gca().patches:
                percentage = f'{100*p.get_height()/total:.1f}%'
                plt.gca().annotate(percentage, (p.get_x()+p.get_width()/2., p.get_height()),
                                 ha='center', va='bottom')
        else:
            df[target].hist(bins=30, alpha=0.7)
            plt.title(f"Distribution of {target}")
            plt.xlabel(target)
            plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _create_correlation_heatmap(df: pd.DataFrame) -> None:
        """Create correlation heatmap for numerical features."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(num_cols) >= 2:
            print(f"\n{'='*20} CORRELATION ANALYSIS {'='*20}")
            
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[num_cols].corr()
            
            # Mask for upper triangle
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap="RdYlBu_r", 
                       center=0, square=True, linewidths=0.5, fmt='.2f')
            plt.title("Feature Correlation Heatmap", fontsize=16, pad=20)
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def _plot_distributions(df: pd.DataFrame) -> None:
        """Plot distributions for numerical features."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(num_cols) > 0:
            print(f"\n{'='*20} FEATURE DISTRIBUTIONS {'='*20}")
            
            # Calculate subplot dimensions
            n_cols = min(4, len(num_cols))
            n_rows = (len(num_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(num_cols):
                if i < len(axes):
                    df[col].hist(bins=30, ax=axes[i], alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(num_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def _analyze_categorical_features(df: pd.DataFrame, max_cat_cardinality: int) -> None:
        """Analyze categorical features."""
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        cat_cols = [col for col in cat_cols if df[col].nunique() <= max_cat_cardinality]
        
        if len(cat_cols) > 0:
            print(f"\n{'='*20} CATEGORICAL FEATURES {'='*20}")
            
            for col in cat_cols:
                print(f"\n{col} - Unique Values: {df[col].nunique()}")
                value_counts = df[col].value_counts()
                print(value_counts.head(10).to_string())
                
                plt.figure(figsize=(10, max(6, len(value_counts)//2)))
                sns.countplot(data=df, y=col, order=value_counts.index)
                plt.title(f"Value Counts: {col}")
                plt.tight_layout()
                plt.show()