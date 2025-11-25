import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
import matplotlib.pyplot as plt
import os
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('energy_forecaster.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and detects anomalies."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: list) -> Tuple[bool, list]:
        """Check for required columns and basic data quality."""
        issues = []
        
        # Check required columns
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for nulls
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Check for infinite values
        numeric_cols = df[required_cols].select_dtypes(include=[np.number]).columns
        inf_counts = np.isinf(df[numeric_cols]).sum()
        if inf_counts.any():
            issues.append(f"Infinite values found: {inf_counts[inf_counts > 0].to_dict()}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def detect_outliers(series: pd.Series, threshold: float = 3.5) -> pd.Series:
        """Detect outliers using modified z-score."""
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / (mad + 1e-10)
        return np.abs(modified_z_scores) > threshold
    
    @staticmethod
    def check_data_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Ensure no temporal overlap between train and test."""
        train_max = train_df.index.max()
        test_min = test_df.index.min()
        
        if train_max >= test_min:
            logger.error(f"DATA LEAKAGE DETECTED: Train ends at {train_max}, test starts at {test_min}")
            return False
        
        logger.info(f"✓ No temporal overlap: Train ends {train_max}, Test starts {test_min}")
        return True


class FeatureEngineer:
    """Handles all feature engineering operations."""
    
    @staticmethod
    def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        df = df.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 20)).astype(int)
        
        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, column: str, lags: list) -> pd.DataFrame:
        """Create lag features safely."""
        df = df.copy()
        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)
        return df
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, column: str, windows: list) -> pd.DataFrame:
        """Create rolling statistics."""
        df = df.copy()
        for window in windows:
            df[f'{column}_roll_mean_{window}h'] = df[column].rolling(window=window, min_periods=1).mean()
            df[f'{column}_roll_std_{window}h'] = df[column].rolling(window=window, min_periods=1).std()
        return df
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        df = df.copy()
        if 'temperature' in df.columns and 'hour' in df.columns:
            df['temp_hour_interaction'] = df['temperature'] * df['hour']
        return df


class EnergyForecaster:
    """Production-grade energy forecasting system."""
    
    def __init__(self, data_path: str = "processed_microgrid_data.csv", 
                 config: Optional[Dict] = None):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.training_metrics = {}
        
        # Default configuration
        self.config = config or {
            'split_ratio': 0.8,
            'cv_splits': 3,
            'random_state': 42,
            'optimize': True,
            'n_iter': 15,  # Increased from 10
            'demand_lags': [1, 2, 24],
            'solar_lags': [1],
            'rolling_windows': [12, 24, 168]  # 12h, 24h, 1 week
        }
        
        self.validator = DataValidator()
        self.feature_engineer = FeatureEngineer()
        
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load data with comprehensive validation."""
        logger.info("=" * 60)
        logger.info("STEP 1: LOADING AND VALIDATING DATA")
        logger.info("=" * 60)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        try:
            df = pd.read_csv(self.data_path, parse_dates=["timestamp"], index_col="timestamp")
            logger.info(f"✓ Loaded {len(df)} records from {self.data_path}")
            logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
            
            # Basic validation
            required_cols = ['demand_kwh', 'generation_kwh']
            is_valid, issues = self.validator.validate_dataframe(df, required_cols)
            
            if not is_valid:
                logger.error(f"Data validation failed: {issues}")
                raise ValueError(f"Data quality issues: {issues}")
            
            # Check for outliers
            for col in required_cols:
                outliers = self.validator.detect_outliers(df[col])
                outlier_pct = (outliers.sum() / len(df)) * 100
                logger.info(f"  {col}: {outliers.sum()} outliers ({outlier_pct:.2f}%)")
                
                if outlier_pct > 5:
                    logger.warning(f"  ⚠️ High outlier percentage in {col}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        logger.info("\nSTEP 2: FEATURE ENGINEERING")
        logger.info("-" * 60)
        
        df = df.copy()
        
        # Temporal features
        df = self.feature_engineer.create_temporal_features(df)
        logger.info("✓ Created temporal features")
        
        # Demand features
        df = self.feature_engineer.create_lag_features(
            df, 'demand_kwh', self.config['demand_lags']
        )
        df = self.feature_engineer.create_rolling_features(
            df, 'demand_kwh', self.config['rolling_windows']
        )
        logger.info(f"✓ Created demand lag and rolling features")
        
        # Solar features
        df = self.feature_engineer.create_lag_features(
            df, 'generation_kwh', self.config['solar_lags']
        )
        logger.info(f"✓ Created solar lag features")
        
        # Interaction features
        df = self.feature_engineer.create_interaction_features(df)
        logger.info("✓ Created interaction features")
        
        # Drop rows with NaN from feature creation
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        logger.info(f"✓ Dropped {dropped} rows with NaN from feature engineering")
        
        return df
    
    def train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological split with validation."""
        split_idx = int(len(df) * self.config['split_ratio'])
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Check for data leakage
        self.validator.check_data_leakage(train_df, test_df)
        
        logger.info(f"\n✓ Train: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
        logger.info(f"✓ Test:  {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")
        
        return train_df, test_df
    
    def get_feature_set(self, model_type: str) -> list:
        """Get feature set for specific model type."""
        base_temporal = ['hour', 'dayofweek', 'is_weekend', 'month', 
                        'hour_sin', 'hour_cos', 'is_night']
        
        if model_type == "demand":
            features = base_temporal.copy()
            
            # Add lag features
            for lag in self.config['demand_lags']:
                features.append(f'demand_kwh_lag_{lag}')
            
            # Add rolling features
            for window in self.config['rolling_windows']:
                features.append(f'demand_kwh_roll_mean_{window}h')
                features.append(f'demand_kwh_roll_std_{window}h')
            
            # Add weather if available
            if 'temperature' in self.training_data.columns:
                features.extend(['temperature', 'temp_hour_interaction'])
            
        elif model_type == "solar":
            features = ['hour', 'month', 'hour_sin', 'hour_cos', 'is_night']
            
            # Add lag features
            for lag in self.config['solar_lags']:
                features.append(f'generation_kwh_lag_{lag}')
            
            # Add weather if available
            weather_cols = ['irradiance', 'cloud_cover', 'temperature']
            for col in weather_cols:
                if col in self.training_data.columns:
                    features.append(col)
        
        return features
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            model_type: str) -> xgb.XGBRegressor:
        """Enhanced hyperparameter tuning."""
        logger.info(f"\n  🔍 Tuning hyperparameters for {model_type.upper()}...")
        
        param_dist = {
            'n_estimators': [500, 1000, 1500, 2000],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6, 7],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
        
        xgb_model = xgb.XGBRegressor(
            random_state=self.config['random_state'],
            n_jobs=-1,
            tree_method='hist'
        )
        
        tscv = TimeSeriesSplit(n_splits=self.config['cv_splits'])
        
        from sklearn.model_selection import RandomizedSearchCV
        
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_dist,
            n_iter=self.config['n_iter'],
            scoring='neg_mean_absolute_error',
            cv=tscv,
            verbose=0,
            n_jobs=-1,
            random_state=self.config['random_state']
        )
        
        random_search.fit(X_train, y_train)
        
        # Log CV results
        cv_mae = -random_search.best_score_
        logger.info(f"  ✓ Best CV MAE: {cv_mae:.4f}")
        logger.info(f"  ✓ Best params: {random_search.best_params_}")
        
        # Store CV metrics
        self.training_metrics[model_type] = {
            'cv_mae': cv_mae,
            'best_params': random_search.best_params_
        }
        
        return random_search.best_estimator_
    
    def calculate_baseline(self, y_train: pd.Series, y_test: pd.Series, 
                          model_type: str) -> Dict:
        """Calculate baseline forecasts for comparison."""
        # Naive forecast (last value)
        naive_pred = np.full(len(y_test), y_train.iloc[-1])
        naive_mae = mean_absolute_error(y_test, naive_pred)
        
        # Seasonal average (same hour of day)
        hour_averages = y_train.groupby(y_train.index.hour).mean()
        seasonal_pred = y_test.index.hour.map(hour_averages)
        seasonal_mae = mean_absolute_error(y_test, seasonal_pred)
        
        return {
            'naive_mae': naive_mae,
            'seasonal_mae': seasonal_mae
        }
    
    def evaluate_comprehensive(self, model: xgb.XGBRegressor, 
                              X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series, 
                              model_name: str) -> Tuple[np.ndarray, Dict]:
        """Comprehensive model evaluation."""
        logger.info(f"\n" + "=" * 60)
        logger.info(f"EVALUATION: {model_name.upper()} MODEL")
        logger.info("=" * 60)
        
        # Predictions
        train_preds = np.maximum(model.predict(X_train), 0)
        test_preds = np.maximum(model.predict(X_test), 0)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_preds)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        
        # MAPE (avoid division by zero)
        test_mape = np.mean(np.abs((y_test - test_preds) / (y_test + 1e-10))) * 100
        
        # Baseline comparison
        baselines = self.calculate_baseline(y_train, y_test, model_name)
        
        logger.info(f"\n📊 Training Set:")
        logger.info(f"   MAE:  {train_mae:.4f} kWh")
        logger.info(f"   RMSE: {train_rmse:.4f} kWh")
        
        logger.info(f"\n📊 Test Set:")
        logger.info(f"   MAE:  {test_mae:.4f} kWh")
        logger.info(f"   RMSE: {test_rmse:.4f} kWh")
        logger.info(f"   MAPE: {test_mape:.2f}%")
        
        logger.info(f"\n📈 Baseline Comparison:")
        logger.info(f"   Naive forecast MAE:    {baselines['naive_mae']:.4f}")
        logger.info(f"   Seasonal avg MAE:      {baselines['seasonal_mae']:.4f}")
        
        improvement_naive = ((baselines['naive_mae'] - test_mae) / baselines['naive_mae']) * 100
        improvement_seasonal = ((baselines['seasonal_mae'] - test_mae) / baselines['seasonal_mae']) * 100
        
        logger.info(f"\n✨ Model Improvement:")
        logger.info(f"   vs Naive:    {improvement_naive:+.1f}%")
        logger.info(f"   vs Seasonal: {improvement_seasonal:+.1f}%")
        
        # Warning flags
        if train_mae < test_mae * 0.5:
            logger.warning("⚠️  Possible overfitting: Train MAE much lower than Test MAE")
        
        if test_mae < 0.01 and model_name == "solar":
            logger.warning("⚠️  Suspiciously low error - check for data leakage!")
        
        metrics = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'baselines': baselines,
            'improvement_vs_naive': improvement_naive,
            'improvement_vs_seasonal': improvement_seasonal
        }
        
        return test_preds, metrics
    
    def plot_feature_importance(self, model: xgb.XGBRegressor, 
                               feature_names: list, title: str, filename: str):
        """Enhanced feature importance plot."""
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        # Calculate cumulative importance
        cumulative = np.cumsum(importance[indices])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        ax1.bar(range(len(indices)), importance[indices], color='#4cafa6', alpha=0.8)
        ax1.set_title(f"Feature Importance: {title}", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Features")
        ax1.set_ylabel("Importance Score")
        ax1.set_xticks(range(len(indices)))
        ax1.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Cumulative importance
        ax2.plot(range(len(indices)), cumulative, marker='o', color='#007acc', linewidth=2)
        ax2.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
        ax2.axhline(y=0.9, color='orange', linestyle='--', label='90% threshold')
        ax2.set_title("Cumulative Feature Importance", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Number of Features")
        ax2.set_ylabel("Cumulative Importance")
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Feature importance saved to {filename}")
        
        # Log top features
        logger.info(f"\n  Top 5 Features for {title}:")
        for i, idx in enumerate(indices[:5], 1):
            logger.info(f"    {i}. {feature_names[idx]}: {importance[idx]:.4f}")
    
    def plot_results(self, y_test: pd.Series, preds: np.ndarray, 
                    title: str, filename: str, metrics: Dict):
        """Enhanced results visualization."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: First 168 hours (1 week)
        limit = min(168, len(y_test))
        axes[0].plot(y_test.index[:limit], y_test.values[:limit], 
                    label="Actual", color='black', alpha=0.7, linewidth=2)
        axes[0].plot(y_test.index[:limit], preds[:limit], 
                    label="Predicted", color='#007acc', linestyle='--', linewidth=2)
        axes[0].set_title(f"Forecast: {title} (First Week)", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Energy (kWh)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        residuals = y_test.values - preds
        axes[1].scatter(preds, residuals, alpha=0.5, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_title("Residual Plot", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Predicted Values (kWh)")
        axes[1].set_ylabel("Residuals (kWh)")
        axes[1].grid(True, alpha=0.3)
        
        # Add metrics text box
        metrics_text = f"MAE: {metrics['test_mae']:.4f} kWh\n"
        metrics_text += f"RMSE: {metrics['test_rmse']:.4f} kWh\n"
        metrics_text += f"MAPE: {metrics['test_mape']:.2f}%"
        axes[0].text(0.02, 0.98, metrics_text, transform=axes[0].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Forecast visualization saved to {filename}")
    
    def save_models(self):
        """Save models with metadata."""
        logger.info("\n" + "=" * 60)
        logger.info("SAVING MODELS")
        logger.info("=" * 60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_type, model in self.models.items():
            filename = f"{model_type}_model_{timestamp}.pkl"
            
            model_package = {
                'model': model,
                'feature_names': self.feature_names[model_type],
                'metrics': self.training_metrics.get(model_type, {}),
                'timestamp': timestamp,
                'config': self.config
            }
            
            joblib.dump(model_package, filename)
            logger.info(f"✓ {model_type.capitalize()} model saved: {filename}")
        
        logger.info("\n✅ All models saved successfully!")
    
    def run(self):
        """Main training pipeline."""
        try:
            # Load and prepare data
            df = self.load_and_validate_data()
            df = self.engineer_features(df)
            
            # Store for feature set creation
            self.training_data = df
            
            train_df, test_df = self.train_test_split(df)
            
            # Train both models
            for model_type in ['demand', 'solar']:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"TRAINING {model_type.upper()} MODEL")
                logger.info(f"{'=' * 60}")
                
                # Get features
                features = self.get_feature_set(model_type)
                self.feature_names[model_type] = features
                
                target_col = 'demand_kwh' if model_type == 'demand' else 'generation_kwh'
                
                # Prepare data
                X_train = train_df[features]
                y_train = train_df[target_col]
                X_test = test_df[features]
                y_test = test_df[target_col]
                
                logger.info(f"\n📋 Feature Set ({len(features)} features):")
                for i, feat in enumerate(features, 1):
                    logger.info(f"   {i}. {feat}")
                
                # Train model
                if self.config['optimize']:
                    model = self.tune_hyperparameters(X_train, y_train, model_type)
                else:
                    model = xgb.XGBRegressor(
                        n_estimators=1000,
                        learning_rate=0.05,
                        max_depth=5,
                        n_jobs=-1,
                        random_state=self.config['random_state']
                    )
                    model.fit(X_train, y_train)
                
                self.models[model_type] = model
                
                # Evaluate
                preds, metrics = self.evaluate_comprehensive(
                    model, X_train, y_train, X_test, y_test, model_type
                )
                
                # Visualize
                self.plot_results(y_test, preds, 
                                f"{model_type.capitalize()}", 
                                f"forecast_{model_type}.png",
                                metrics)
                self.plot_feature_importance(model, features, 
                                           model_type.capitalize(),
                                           f"importance_{model_type}.png")
            
            # Save models
            self.save_models()
            
            logger.info("\n" + "=" * 60)
            logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"\n❌ Training failed: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    # Configuration
    config = {
        'split_ratio': 0.8,
        'cv_splits': 3,
        'random_state': 42,
        'optimize': True,
        'n_iter': 15,
        'demand_lags': [1, 2, 24, 168],  # 1h, 2h, 1day, 1week
        'solar_lags': [1],
        'rolling_windows': [12, 24, 168]
    }
    
    if not os.path.exists("processed_microgrid_data.csv"):
        logger.error("⚠️  'processed_microgrid_data.csv' not found!")
    else:
        forecaster = EnergyForecaster(config=config)
        forecaster.run()