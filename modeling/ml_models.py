"""
Machine learning models for water stress prediction
Includes Random Forest, XGBoost, and ensemble methods
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class WaterStressMLPipeline:
    """Machine learning pipeline for water stress prediction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def prepare_features(self, df: pd.DataFrame, 
                        target_column: str = 'water_stress_level',
                        lag_features: List[int] = [1, 3, 7, 14]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning
        
        Args:
            df: Input DataFrame with time series data
            target_column: Name of target variable
            lag_features: List of lag periods for creating lagged features
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            df = df.copy().sort_values('date')
            
            # Select relevant feature columns
            feature_columns = [
                'NDVI', 'GNDVI', 'NDWI_Gao', 'EVI', 'SAVI', 'LAI',
                'T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'PET_PT_JPL',
                'VPD', 'GDD', 'water_balance_daily', 'water_balance_7d', 'water_balance_30d',
                'NDVI_trend_slope', 'NDVI_anomaly_std', 'CWSI'
            ]
            
            # Filter available columns
            available_features = [col for col in feature_columns if col in df.columns]
            features_df = df[available_features + ['date']].copy()
            
            # Add time-based features
            features_df = self._add_time_features(features_df)
            
            # Add lagged features
            features_df = self._add_lag_features(features_df, available_features, lag_features)
            
            # Add rolling statistics
            features_df = self._add_rolling_features(features_df, available_features)
            
            # Add interaction features
            features_df = self._add_interaction_features(features_df)
            
            # Create target variable if not exists
            if target_column not in df.columns:
                features_df = self._create_target_variable(features_df, df)
                target_column = 'water_stress_level'
            else:
                features_df[target_column] = df[target_column]
            
            # Remove rows with missing target
            mask = features_df[target_column].notna()
            features_df = features_df[mask]
            
            # Separate features and target
            feature_cols = [col for col in features_df.columns if col not in ['date', target_column]]
            X = features_df[feature_cols]
            y = features_df[target_column]
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            self.logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            raise
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_year'] = df['date'].dt.dayofyear
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['season'] = df['month'].map({12: 1, 1: 1, 2: 1,  # Summer (Southern Hemisphere)
                                          3: 2, 4: 2, 5: 2,   # Autumn
                                          6: 3, 7: 3, 8: 3,   # Winter
                                          9: 4, 10: 4, 11: 4}) # Spring
            
            # Cyclical encoding for temporal features
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding time features: {e}")
            return df
    
    def _add_lag_features(self, df: pd.DataFrame, 
                         feature_columns: List[str], 
                         lag_periods: List[int]) -> pd.DataFrame:
        """Add lagged features"""
        try:
            for col in feature_columns:
                if col in df.columns:
                    for lag in lag_periods:
                        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding lag features: {e}")
            return df
    
    def _add_rolling_features(self, df: pd.DataFrame, 
                             feature_columns: List[str],
                             windows: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
        """Add rolling statistical features"""
        try:
            for col in feature_columns:
                if col in df.columns:
                    for window in windows:
                        df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                        df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                        df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                        df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding rolling features: {e}")
            return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between key variables"""
        try:
            # Vegetation x Weather interactions
            if all(col in df.columns for col in ['NDVI', 'T2M']):
                df['NDVI_x_temp'] = df['NDVI'] * df['T2M']
            
            if all(col in df.columns for col in ['NDVI', 'VPD']):
                df['NDVI_x_VPD'] = df['NDVI'] * df['VPD']
            
            if all(col in df.columns for col in ['NDWI_Gao', 'PRECTOTCORR']):
                df['NDWI_x_precip'] = df['NDWI_Gao'] * df['PRECTOTCORR']
            
            # Water balance ratios
            if all(col in df.columns for col in ['PRECTOTCORR', 'PET_PT_JPL']):
                df['precip_pet_ratio'] = df['PRECTOTCORR'] / (df['PET_PT_JPL'] + 0.1)
            
            # Stress indicators combination
            if all(col in df.columns for col in ['NDVI', 'NDWI_Gao']):
                df['vegetation_water_index'] = df['NDVI'] * df['NDWI_Gao']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding interaction features: {e}")
            return df
    
    def _create_target_variable(self, features_df: pd.DataFrame, 
                               original_df: pd.DataFrame) -> pd.DataFrame:
        """Create water stress target variable from multiple indicators"""
        try:
            # Define stress thresholds
            stress_indicators = []
            
            # NDVI-based stress
            if 'NDVI' in features_df.columns:
                ndvi_stress = (features_df['NDVI'] < 0.6).astype(int)
                stress_indicators.append(ndvi_stress)
            
            # NDWI-based stress
            if 'NDWI_Gao' in features_df.columns:
                ndwi_stress = (features_df['NDWI_Gao'] < 0.3).astype(int)
                stress_indicators.append(ndwi_stress)
            
            # Temperature stress
            if 'T2M_MAX' in features_df.columns:
                temp_stress = (features_df['T2M_MAX'] > 35).astype(int)
                stress_indicators.append(temp_stress)
            
            # Water balance stress
            if 'water_balance_7d' in features_df.columns:
                wb_stress = (features_df['water_balance_7d'] < -20).astype(int)
                stress_indicators.append(wb_stress)
            
            # CWSI stress
            if 'CWSI' in features_df.columns:
                cwsi_stress = (features_df['CWSI'] > 0.4).astype(int)
                stress_indicators.append(cwsi_stress)
            
            # Combine stress indicators
            if stress_indicators:
                combined_stress = np.sum(stress_indicators, axis=0)
                
                # Create categorical stress levels
                features_df['water_stress_level'] = np.where(
                    combined_stress >= 3, 2,  # High stress
                    np.where(combined_stress >= 1, 1, 0)  # Moderate stress, No stress
                )
            else:
                # Default to no stress if no indicators available
                features_df['water_stress_level'] = 0
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating target variable: {e}")
            return features_df
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series,
                           task_type: str = 'classification',
                           test_size: float = 0.2,
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train Random Forest model
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: 'classification' or 'regression'
            test_size: Proportion of test set
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with model and performance metrics
        """
        try:
            # Split data with temporal consideration
            if 'date' in X.columns:
                X_train, X_test, y_train, y_test = self._temporal_split(X, y, test_size)
                X_train = X_train.drop('date', axis=1)
                X_test = X_test.drop('date', axis=1)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y if task_type == 'classification' else None
                )
            
            # Choose model based on task type
            if task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate performance metrics
            if task_type == 'classification':
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                performance = {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'classification_report': classification_report(y_test, y_pred_test),
                    'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
                }
            else:
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                performance = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                }
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
            performance['cv_mean'] = cv_scores.mean()
            performance['cv_std'] = cv_scores.std()
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Store model components
            self.models['random_forest'] = model
            self.scalers['random_forest'] = scaler
            self.feature_importance['random_forest'] = feature_importance
            self.model_performance['random_forest'] = performance
            
            self.logger.info(f"Random Forest trained successfully. Test score: {test_score:.4f}")
            
            return {
                'model': model,
                'scaler': scaler,
                'performance': performance,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest: {e}")
            raise
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series,
                     task_type: str = 'classification',
                     test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train XGBoost model
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: 'classification' or 'regression'
            test_size: Proportion of test set
            
        Returns:
            Dictionary with model and performance metrics
        """
        try:
            # Split data
            if 'date' in X.columns:
                X_train, X_test, y_train, y_test = self._temporal_split(X, y, test_size)
                X_train = X_train.drop('date', axis=1)
                X_test = X_test.drop('date', axis=1)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            # Choose model based on task type
            if task_type == 'classification':
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss' if len(np.unique(y)) > 2 else 'error'
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='rmse'
                )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train with early stopping
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate performance metrics
            if task_type == 'classification':
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                performance = {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'classification_report': classification_report(y_test, y_pred_test),
                    'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
                }
            else:
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                performance = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                }
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Store model components
            self.models['xgboost'] = model
            self.scalers['xgboost'] = scaler
            self.feature_importance['xgboost'] = feature_importance
            self.model_performance['xgboost'] = performance
            
            self.logger.info(f"XGBoost trained successfully. Test score: {test_score:.4f}")
            
            return {
                'model': model,
                'scaler': scaler,
                'performance': performance,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost: {e}")
            raise
    
    def _temporal_split(self, X: pd.DataFrame, y: pd.Series, 
                       test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data temporally (earlier data for training, later for testing)"""
        try:
            if 'date' not in X.columns:
                return train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Sort by date
            sorted_indices = X['date'].argsort()
            X_sorted = X.iloc[sorted_indices]
            y_sorted = y.iloc[sorted_indices]
            
            # Split temporally
            split_idx = int(len(X_sorted) * (1 - test_size))
            
            X_train = X_sorted.iloc[:split_idx]
            X_test = X_sorted.iloc[split_idx:]
            y_train = y_sorted.iloc[:split_idx]
            y_test = y_sorted.iloc[split_idx:]
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error in temporal split: {e}")
            return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series,
                             model_type: str = 'random_forest',
                             task_type: str = 'classification') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: 'random_forest' or 'xgboost'
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with best model and parameters
        """
        try:
            # Define parameter grids
            if model_type == 'random_forest':
                if task_type == 'classification':
                    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 15, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                else:
                    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 15, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
            
            elif model_type == 'xgboost':
                if task_type == 'classification':
                    base_model = xgb.XGBClassifier(random_state=42)
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                else:
                    base_model = xgb.XGBRegressor(random_state=42)
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0]
                    }
            
            # Prepare data
            X_clean = X.drop('date', axis=1) if 'date' in X.columns else X
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Grid search
            scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
            grid_search = GridSearchCV(
                base_model, param_grid, cv=tscv, scoring=scoring, n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_scaled, y)
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            self.logger.info(f"Best parameters for {model_type}: {best_params}")
            self.logger.info(f"Best cross-validation score: {best_score:.4f}")
            
            return {
                'best_model': best_model,
                'best_params': best_params,
                'best_score': best_score,
                'scaler': scaler,
                'grid_search': grid_search
            }
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter tuning: {e}")
            raise
    
    def predict_water_stress(self, X_new: pd.DataFrame,
                           model_name: str = 'random_forest') -> np.ndarray:
        """
        Predict water stress for new data
        
        Args:
            X_new: New feature data
            model_name: Name of trained model to use
            
        Returns:
            Predictions array
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            
            model = self.models[model_name]
            scaler = self.scalers[model_name]
            
            # Prepare features
            X_clean = X_new.drop('date', axis=1) if 'date' in X_new.columns else X_new
            X_scaled = scaler.transform(X_clean.fillna(0))
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
            # Get prediction probabilities for classification
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)
                return predictions, probabilities
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            raise
    
    def save_models(self, filepath: str):
        """Save trained models and components"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance,
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self, filepath: str):
        """Load trained models and components"""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_importance = model_data['feature_importance']
            self.model_performance = model_data['model_performance']
            
            self.logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise