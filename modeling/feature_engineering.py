"""
Feature engineering for water stress detection models
Creates and transforms features for machine learning
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Any
import logging

class FeatureEngineering:
    """Feature engineering pipeline for water stress detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.feature_selectors = {}
        self.transformers = {}
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive temporal features
        
        Args:
            df: DataFrame with date column
            
        Returns:
            DataFrame with temporal features
        """
        try:
            df = df.copy()
            
            if 'date' not in df.columns:
                return df
            
            # Basic temporal features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_year'] = df['date'].dt.dayofyear
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['quarter'] = df['date'].dt.quarter
            df['is_weekend'] = df['date'].dt.weekday.isin([5, 6]).astype(int)
            
            # Argentine agricultural seasons (Southern Hemisphere)
            df['season'] = df['month'].map({
                12: 'summer', 1: 'summer', 2: 'summer',
                3: 'autumn', 4: 'autumn', 5: 'autumn',
                6: 'winter', 7: 'winter', 8: 'winter',
                9: 'spring', 10: 'spring', 11: 'spring'
            })
            
            # Growing season indicators for major crops
            df['soybean_season'] = df['month'].isin([10, 11, 12, 1, 2, 3, 4]).astype(int)
            df['corn_season'] = df['month'].isin([9, 10, 11, 12, 1, 2, 3]).astype(int)
            df['wheat_season'] = df['month'].isin([5, 6, 7, 8, 9, 10, 11, 12]).astype(int)
            
            # Cyclical encoding for temporal features
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
            df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
            
            # Days since epoch (for trend features)
            epoch = pd.Timestamp('2000-01-01')
            df['days_since_epoch'] = (df['date'] - epoch).dt.days
            
            # Agricultural calendar features
            df = self._add_agricultural_calendar_features(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating temporal features: {e}")
            return df
    
    def _add_agricultural_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add agricultural calendar-specific features"""
        try:
            # Planting and harvest periods for Argentina
            df['soybean_planting'] = df['month'].isin([10, 11, 12]).astype(int)
            df['soybean_flowering'] = df['month'].isin([12, 1]).astype(int)
            df['soybean_harvest'] = df['month'].isin([3, 4]).astype(int)
            
            df['corn_planting'] = df['month'].isin([9, 10, 11]).astype(int)
            df['corn_flowering'] = df['month'].isin([12, 1]).astype(int)
            df['corn_harvest'] = df['month'].isin([2, 3]).astype(int)
            
            df['wheat_planting'] = df['month'].isin([5, 6, 7]).astype(int)
            df['wheat_flowering'] = df['month'].isin([9, 10]).astype(int)
            df['wheat_harvest'] = df['month'].isin([11, 12]).astype(int)
            
            # Critical growth periods (water stress sensitive)
            df['critical_period_soybean'] = df['month'].isin([12, 1, 2]).astype(int)
            df['critical_period_corn'] = df['month'].isin([12, 1]).astype(int)
            df['critical_period_wheat'] = df['month'].isin([9, 10, 11]).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding agricultural calendar features: {e}")
            return df
    
    def create_vegetation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced vegetation-based features
        
        Args:
            df: DataFrame with vegetation indices
            
        Returns:
            DataFrame with vegetation features
        """
        try:
            df = df.copy()
            
            vegetation_indices = ['NDVI', 'GNDVI', 'NDWI_Gao', 'EVI', 'SAVI', 'LAI']
            
            for vi in vegetation_indices:
                if vi not in df.columns:
                    continue
                
                # Rate of change
                df[f'{vi}_change_rate'] = df[vi].diff()
                df[f'{vi}_change_rate_3d'] = df[vi].diff(3)
                df[f'{vi}_change_rate_7d'] = df[vi].diff(7)
                
                # Acceleration (second derivative)
                df[f'{vi}_acceleration'] = df[f'{vi}_change_rate'].diff()
                
                # Relative change
                df[f'{vi}_pct_change'] = df[vi].pct_change()
                df[f'{vi}_pct_change_7d'] = df[vi].pct_change(7)
                
                # Moving averages and deviations
                for window in [3, 7, 14, 30]:
                    df[f'{vi}_ma_{window}'] = df[vi].rolling(window).mean()
                    df[f'{vi}_std_{window}'] = df[vi].rolling(window).std()
                    df[f'{vi}_deviation_{window}'] = df[vi] - df[f'{vi}_ma_{window}']
                    df[f'{vi}_cv_{window}'] = df[f'{vi}_std_{window}'] / df[f'{vi}_ma_{window}']
                
                # Percentile rankings
                df[f'{vi}_percentile_30d'] = df[vi].rolling(30).rank(pct=True)
                df[f'{vi}_percentile_90d'] = df[vi].rolling(90).rank(pct=True)
                
                # Seasonal comparisons
                if 'day_of_year' in df.columns:
                    df = self._add_seasonal_comparisons(df, vi)
            
            # Vegetation index ratios and combinations
            df = self._create_vegetation_ratios(df)
            
            # Vegetation stress indicators
            df = self._create_vegetation_stress_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating vegetation features: {e}")
            return df
    
    def _add_seasonal_comparisons(self, df: pd.DataFrame, vi: str) -> pd.DataFrame:
        """Add seasonal comparison features for vegetation indices"""
        try:
            # Calculate historical mean for each day of year (±7 day window)
            df['temp_day_of_year'] = df['day_of_year']
            
            seasonal_stats = []
            for doy in range(1, 367):
                # Get ±7 day window (circular)
                doy_range = [(doy + i - 7) % 366 + 1 for i in range(15)]
                
                historical_data = df[df['temp_day_of_year'].isin(doy_range)][vi].dropna()
                
                if len(historical_data) > 5:
                    seasonal_stats.append({
                        'day_of_year': doy,
                        f'{vi}_seasonal_mean': historical_data.mean(),
                        f'{vi}_seasonal_std': historical_data.std()
                    })
            
            if seasonal_stats:
                seasonal_df = pd.DataFrame(seasonal_stats)
                df = df.merge(seasonal_df, on='day_of_year', how='left')
                
                # Calculate anomalies
                df[f'{vi}_seasonal_anomaly'] = df[vi] - df[f'{vi}_seasonal_mean']
                df[f'{vi}_seasonal_zscore'] = (
                    df[f'{vi}_seasonal_anomaly'] / df[f'{vi}_seasonal_std']
                )
            
            df = df.drop('temp_day_of_year', axis=1)
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding seasonal comparisons: {e}")
            return df
    
    def _create_vegetation_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create vegetation index ratios and combinations"""
        try:
            # NDVI-based ratios
            if all(col in df.columns for col in ['NDVI', 'EVI']):
                df['NDVI_EVI_ratio'] = df['NDVI'] / (df['EVI'] + 0.001)
            
            if all(col in df.columns for col in ['NDVI', 'SAVI']):
                df['NDVI_SAVI_ratio'] = df['NDVI'] / (df['SAVI'] + 0.001)
            
            # Water-vegetation relationships
            if all(col in df.columns for col in ['NDVI', 'NDWI_Gao']):
                df['vegetation_water_balance'] = df['NDVI'] / (df['NDWI_Gao'] + 0.5)
                df['ndvi_ndwi_product'] = df['NDVI'] * df['NDWI_Gao']
                df['ndvi_ndwi_diff'] = df['NDVI'] - df['NDWI_Gao']
            
            # Green-red vegetation relationships
            if all(col in df.columns for col in ['NDVI', 'GNDVI']):
                df['green_red_balance'] = df['GNDVI'] / (df['NDVI'] + 0.001)
            
            # Multi-index vegetation vigor
            vi_cols = [col for col in df.columns if col in ['NDVI', 'EVI', 'SAVI', 'GNDVI']]
            if len(vi_cols) >= 2:
                df['vegetation_vigor_index'] = df[vi_cols].mean(axis=1)
                df['vegetation_vigor_std'] = df[vi_cols].std(axis=1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating vegetation ratios: {e}")
            return df
    
    def _create_vegetation_stress_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create vegetation stress indicator features"""
        try:
            # NDVI stress levels
            if 'NDVI' in df.columns:
                df['ndvi_stress_mild'] = (df['NDVI'] < 0.7).astype(int)
                df['ndvi_stress_moderate'] = (df['NDVI'] < 0.6).astype(int)
                df['ndvi_stress_severe'] = (df['NDVI'] < 0.4).astype(int)
                
                # Sustained stress periods
                df['ndvi_stress_consecutive'] = (
                    df['ndvi_stress_moderate'].groupby(
                        (df['ndvi_stress_moderate'] != df['ndvi_stress_moderate'].shift()).cumsum()
                    ).cumsum() * df['ndvi_stress_moderate']
                )
            
            # NDWI stress levels
            if 'NDWI_Gao' in df.columns:
                df['ndwi_stress_mild'] = (df['NDWI_Gao'] < 0.4).astype(int)
                df['ndwi_stress_moderate'] = (df['NDWI_Gao'] < 0.3).astype(int)
                df['ndwi_stress_severe'] = (df['NDWI_Gao'] < 0.2).astype(int)
            
            # Vegetation decline indicators
            if 'NDVI_change_rate_7d' in df.columns:
                df['vegetation_declining'] = (df['NDVI_change_rate_7d'] < -0.05).astype(int)
                df['vegetation_rapidly_declining'] = (df['NDVI_change_rate_7d'] < -0.1).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating vegetation stress indicators: {e}")
            return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced weather-based features
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with weather features
        """
        try:
            df = df.copy()
            
            weather_vars = ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'PET_PT_JPL', 'VPD']
            
            for var in weather_vars:
                if var not in df.columns:
                    continue
                
                # Cumulative features
                if var in ['PRECTOTCORR', 'PET_PT_JPL']:
                    for period in [7, 14, 30, 60, 90]:
                        df[f'{var}_cumsum_{period}d'] = df[var].rolling(period).sum()
                
                # Moving statistics
                for window in [3, 7, 14, 30]:
                    df[f'{var}_mean_{window}d'] = df[var].rolling(window).mean()
                    df[f'{var}_max_{window}d'] = df[var].rolling(window).max()
                    df[f'{var}_min_{window}d'] = df[var].rolling(window).min()
                    df[f'{var}_std_{window}d'] = df[var].rolling(window).std()
                
                # Extremes and thresholds
                if var == 'T2M_MAX':
                    for threshold in [30, 32, 35, 38]:
                        df[f'days_above_{threshold}C'] = (df[var] > threshold).astype(int)
                        df[f'consecutive_days_above_{threshold}C'] = self._consecutive_days(
                            df[var] > threshold
                        )
                
                if var == 'PRECTOTCORR':
                    df['dry_day'] = (df[var] <= 0.1).astype(int)
                    df['consecutive_dry_days'] = self._consecutive_days(df['dry_day'])
                    df['heavy_rain'] = (df[var] > 20).astype(int)
            
            # Derived weather features
            df = self._create_weather_derived_features(df)
            
            # Water balance features
            df = self._create_water_balance_features(df)
            
            # Heat stress features
            df = self._create_heat_stress_features(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating weather features: {e}")
            return df
    
    def _consecutive_days(self, condition: pd.Series) -> pd.Series:
        """Calculate consecutive days meeting condition"""
        try:
            groups = (condition != condition.shift()).cumsum()
            return condition.groupby(groups).cumsum() * condition
        except:
            return pd.Series(0, index=condition.index)
    
    def _create_weather_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived weather features"""
        try:
            # Diurnal temperature range
            if all(col in df.columns for col in ['T2M_MAX', 'T2M_MIN']):
                df['diurnal_temp_range'] = df['T2M_MAX'] - df['T2M_MIN']
                df['temp_range_7d_mean'] = df['diurnal_temp_range'].rolling(7).mean()
            
            # Growing degree days (multiple base temperatures)
            if all(col in df.columns for col in ['T2M_MAX', 'T2M_MIN']):
                for base_temp in [8, 10, 12, 15]:
                    df[f'gdd_base_{base_temp}'] = np.maximum(
                        (df['T2M_MAX'] + df['T2M_MIN']) / 2 - base_temp, 0
                    )
                    df[f'gdd_cumsum_{base_temp}'] = df[f'gdd_base_{base_temp}'].cumsum()
            
            # Vapor pressure deficit categories
            if 'VPD' in df.columns:
                df['vpd_low'] = (df['VPD'] < 1.0).astype(int)
                df['vpd_moderate'] = ((df['VPD'] >= 1.0) & (df['VPD'] < 2.0)).astype(int)
                df['vpd_high'] = ((df['VPD'] >= 2.0) & (df['VPD'] < 3.0)).astype(int)
                df['vpd_extreme'] = (df['VPD'] >= 3.0).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating derived weather features: {e}")
            return df
    
    def _create_water_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create water balance features"""
        try:
            if all(col in df.columns for col in ['PRECTOTCORR', 'PET_PT_JPL']):
                df['daily_water_balance'] = df['PRECTOTCORR'] - df['PET_PT_JPL']
                
                # Water balance categories
                df['water_deficit'] = (df['daily_water_balance'] < -2).astype(int)
                df['water_surplus'] = (df['daily_water_balance'] > 5).astype(int)
                
                # Cumulative water balance
                df['cumulative_water_balance'] = df['daily_water_balance'].cumsum()
                
                # Water balance ratios
                df['precip_pet_ratio'] = df['PRECTOTCORR'] / (df['PET_PT_JPL'] + 0.1)
                
                # Standardized water balance (anomalies)
                for period in [30, 60, 90]:
                    rolling_mean = df['daily_water_balance'].rolling(period, center=True).mean()
                    rolling_std = df['daily_water_balance'].rolling(period, center=True).std()
                    df[f'water_balance_zscore_{period}d'] = (
                        (df['daily_water_balance'] - rolling_mean) / rolling_std
                    )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating water balance features: {e}")
            return df
    
    def _create_heat_stress_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create heat stress features"""
        try:
            if 'T2M_MAX' in df.columns:
                # Heat stress thresholds for different crops
                crop_thresholds = {
                    'soybean': 32,
                    'corn': 35,
                    'wheat': 30
                }
                
                for crop, threshold in crop_thresholds.items():
                    df[f'{crop}_heat_stress'] = (df['T2M_MAX'] > threshold).astype(int)
                    
                    # Heat stress duration
                    df[f'{crop}_heat_stress_consecutive'] = self._consecutive_days(
                        df[f'{crop}_heat_stress']
                    )
                
                # Heat units above threshold
                for threshold in [30, 32, 35]:
                    excess_heat = np.maximum(df['T2M_MAX'] - threshold, 0)
                    df[f'heat_units_above_{threshold}'] = excess_heat
                    df[f'heat_units_cumsum_above_{threshold}'] = excess_heat.cumsum()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating heat stress features: {e}")
            return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       method: str = 'univariate',
                       k: int = 50,
                       task_type: str = 'classification') -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the most relevant features
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method ('univariate', 'rfe', 'correlation')
            k: Number of features to select
            task_type: 'classification' or 'regression'
            
        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        try:
            X_clean = X.select_dtypes(include=[np.number]).fillna(0)
            
            if method == 'univariate':
                score_func = f_classif if task_type == 'classification' else f_regression
                selector = SelectKBest(score_func=score_func, k=min(k, X_clean.shape[1]))
                X_selected = selector.fit_transform(X_clean, y)
                selected_features = X_clean.columns[selector.get_support()].tolist()
                
            elif method == 'rfe':
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                if task_type == 'classification':
                    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                else:
                    estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                
                selector = RFE(estimator=estimator, n_features_to_select=min(k, X_clean.shape[1]))
                X_selected = selector.fit_transform(X_clean, y)
                selected_features = X_clean.columns[selector.get_support()].tolist()
                
            elif method == 'correlation':
                # Remove highly correlated features
                corr_matrix = X_clean.corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                # Find features with correlation > 0.95
                to_drop = [column for column in upper_triangle.columns 
                          if any(upper_triangle[column] > 0.95)]
                
                X_selected = X_clean.drop(columns=to_drop)
                selected_features = X_selected.columns.tolist()
                
                # If still too many features, use univariate selection
                if len(selected_features) > k:
                    score_func = f_classif if task_type == 'classification' else f_regression
                    selector = SelectKBest(score_func=score_func, k=k)
                    X_selected = selector.fit_transform(X_selected, y)
                    selected_features = [selected_features[i] for i in selector.get_support(indices=True)]
            
            self.feature_selectors[method] = selector if method != 'correlation' else None
            
            X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            self.logger.info(f"Selected {len(selected_features)} features using {method} method")
            
            return X_selected_df, selected_features
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return X, X.columns.tolist()
    
    def scale_features(self, X: pd.DataFrame, 
                      method: str = 'standard',
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale features using specified method
        
        Args:
            X: Feature matrix
            method: Scaling method ('standard', 'minmax', 'robust')
            fit: Whether to fit scaler or use existing one
            
        Returns:
            Scaled feature matrix
        """
        try:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            X_numeric = X.select_dtypes(include=[np.number])
            
            if fit:
                X_scaled = scaler.fit_transform(X_numeric)
                self.scalers[method] = scaler
            else:
                if method not in self.scalers:
                    raise ValueError(f"Scaler for {method} not fitted yet")
                X_scaled = self.scalers[method].transform(X_numeric)
            
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_numeric.columns, index=X.index)
            
            # Add back non-numeric columns
            non_numeric = X.select_dtypes(exclude=[np.number])
            if not non_numeric.empty:
                X_scaled_df = pd.concat([X_scaled_df, non_numeric], axis=1)
            
            return X_scaled_df
            
        except Exception as e:
            self.logger.error(f"Error scaling features: {e}")
            return X
    
    def create_polynomial_features(self, X: pd.DataFrame, 
                                  columns: List[str],
                                  degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for specified columns"""
        try:
            from sklearn.preprocessing import PolynomialFeatures
            
            X_subset = X[columns]
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
            X_poly = poly.fit_transform(X_subset)
            
            # Create feature names
            feature_names = poly.get_feature_names_out(columns)
            
            # Create DataFrame
            X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
            
            # Remove original columns and add polynomial features
            X_result = X.drop(columns=columns)
            X_result = pd.concat([X_result, X_poly_df], axis=1)
            
            self.transformers['polynomial'] = poly
            
            return X_result
            
        except Exception as e:
            self.logger.error(f"Error creating polynomial features: {e}")
            return X