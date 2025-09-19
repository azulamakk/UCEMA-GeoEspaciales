"""
Time series analysis for water stress detection
Includes trend detection, anomaly identification, and seasonal decomposition
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging

warnings.filterwarnings('ignore')

class TimeSeriesAnalysis:
    """Time series analysis for vegetation and climate data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_trends(self, df: pd.DataFrame, 
                     columns: List[str], 
                     window_size: int = 30) -> pd.DataFrame:
        """
        Detect trends in time series data using multiple methods
        
        Args:
            df: DataFrame with time series data
            columns: List of columns to analyze
            window_size: Window size for trend calculation
            
        Returns:
            DataFrame with trend indicators
        """
        try:
            df = df.copy()
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                # Linear trend (slope)
                df[f'{col}_trend_slope'] = df[col].rolling(window=window_size).apply(
                    lambda x: self._calculate_trend_slope(x), raw=False
                )
                
                # Mann-Kendall trend test
                df[f'{col}_mk_trend'] = df[col].rolling(window=window_size).apply(
                    lambda x: self._mann_kendall_trend(x), raw=False
                )
                
                # Relative change
                df[f'{col}_rel_change'] = df[col].pct_change(periods=window_size)
                
                # Trend direction
                df[f'{col}_trend_direction'] = np.where(
                    df[f'{col}_trend_slope'] > 0.001, 1,  # Increasing
                    np.where(df[f'{col}_trend_slope'] < -0.001, -1, 0)  # Decreasing or stable
                )
                
                # Trend significance
                df[f'{col}_trend_significant'] = (
                    np.abs(df[f'{col}_mk_trend']) > 1.96
                ).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error detecting trends: {e}")
            return df
    
    def _calculate_trend_slope(self, series: pd.Series) -> float:
        """Calculate linear trend slope"""
        try:
            if len(series) < 3 or series.isna().all():
                return np.nan
                
            x = np.arange(len(series))
            y = series.values
            
            # Remove NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 3:
                return np.nan
                
            slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
            return slope
            
        except Exception:
            return np.nan
    
    def _mann_kendall_trend(self, series: pd.Series) -> float:
        """Mann-Kendall trend test statistic"""
        try:
            values = series.dropna().values
            n = len(values)
            
            if n < 3:
                return np.nan
            
            # Calculate S statistic
            s = 0
            for i in range(n-1):
                for j in range(i+1, n):
                    s += np.sign(values[j] - values[i])
            
            # Calculate variance
            var_s = n * (n - 1) * (2 * n + 5) / 18
            
            # Calculate Z statistic
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0
                
            return z
            
        except Exception:
            return np.nan
    
    def detect_anomalies(self, df: pd.DataFrame, 
                        columns: List[str],
                        method: str = 'iqr',
                        threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect anomalies in time series data
        
        Args:
            df: DataFrame with time series data
            columns: List of columns to analyze
            method: Anomaly detection method ('iqr', 'zscore', 'isolation')
            threshold: Threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly flags
        """
        try:
            df = df.copy()
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                if method == 'iqr':
                    df[f'{col}_anomaly'] = self._detect_iqr_anomalies(df[col], threshold)
                elif method == 'zscore':
                    df[f'{col}_anomaly'] = self._detect_zscore_anomalies(df[col], threshold)
                elif method == 'isolation':
                    df[f'{col}_anomaly'] = self._detect_isolation_anomalies(df[[col]])
                
                # Rolling anomaly detection
                df[f'{col}_rolling_anomaly'] = self._detect_rolling_anomalies(df[col])
                
                # Seasonal anomaly detection
                if 'date' in df.columns:
                    df[f'{col}_seasonal_anomaly'] = self._detect_seasonal_anomalies(
                        df[col], df['date']
                    )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return df
    
    def _detect_iqr_anomalies(self, series: pd.Series, threshold: float = 1.5) -> pd.Series:
        """Detect anomalies using Interquartile Range method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            anomalies = (series < lower_bound) | (series > upper_bound)
            return anomalies.astype(int)
            
        except Exception:
            return pd.Series(0, index=series.index)
    
    def _detect_zscore_anomalies(self, series: pd.Series, threshold: float = 2.0) -> pd.Series:
        """Detect anomalies using Z-score method"""
        try:
            z_scores = np.abs(stats.zscore(series.dropna()))
            
            anomalies = pd.Series(0, index=series.index)
            valid_indices = series.dropna().index
            anomalies.loc[valid_indices] = (z_scores > threshold).astype(int)
            
            return anomalies
            
        except Exception:
            return pd.Series(0, index=series.index)
    
    def _detect_isolation_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """Detect anomalies using Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Remove NaN values
            clean_data = df.dropna()
            if len(clean_data) < 10:
                return pd.Series(0, index=df.index)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(clean_data)
            
            # Convert to binary (1 for anomaly, 0 for normal)
            anomalies = pd.Series(0, index=df.index)
            anomalies.loc[clean_data.index] = (anomaly_labels == -1).astype(int)
            
            return anomalies
            
        except Exception:
            return pd.Series(0, index=df.index)
    
    def _detect_rolling_anomalies(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Detect anomalies using rolling statistics"""
        try:
            rolling_mean = series.rolling(window=window, center=True).mean()
            rolling_std = series.rolling(window=window, center=True).std()
            
            z_scores = np.abs((series - rolling_mean) / rolling_std)
            anomalies = (z_scores > 2.5).astype(int)
            
            return anomalies
            
        except Exception:
            return pd.Series(0, index=series.index)
    
    def _detect_seasonal_anomalies(self, series: pd.Series, dates: pd.Series) -> pd.Series:
        """Detect seasonal anomalies"""
        try:
            df_temp = pd.DataFrame({'value': series, 'date': dates})
            df_temp['day_of_year'] = df_temp['date'].dt.dayofyear
            
            # Calculate seasonal statistics
            seasonal_stats = df_temp.groupby('day_of_year')['value'].agg(['mean', 'std'])
            
            # Merge back
            df_temp = df_temp.merge(seasonal_stats, left_on='day_of_year', right_index=True)
            
            # Calculate seasonal Z-scores
            seasonal_z = np.abs((df_temp['value'] - df_temp['mean']) / df_temp['std'])
            anomalies = (seasonal_z > 2.0).astype(int)
            
            return pd.Series(anomalies.values, index=series.index)
            
        except Exception:
            return pd.Series(0, index=series.index)
    
    def seasonal_decomposition(self, df: pd.DataFrame, 
                              columns: List[str],
                              period: int = 365,
                              model: str = 'additive') -> pd.DataFrame:
        """
        Perform seasonal decomposition of time series
        
        Args:
            df: DataFrame with time series data
            columns: List of columns to decompose
            period: Seasonal period (days)
            model: 'additive' or 'multiplicative'
            
        Returns:
            DataFrame with decomposed components
        """
        try:
            df = df.copy().sort_values('date')
            
            for col in columns:
                if col not in df.columns or df[col].isna().all():
                    continue
                
                # Prepare time series
                ts = df.set_index('date')[col].dropna()
                
                if len(ts) < 2 * period:
                    self.logger.warning(f"Insufficient data for seasonal decomposition of {col}")
                    continue
                
                # Resample to daily frequency if needed
                ts = ts.resample('D').interpolate(method='linear')
                
                try:
                    # Perform decomposition
                    decomposition = seasonal_decompose(
                        ts, model=model, period=period, extrapolate_trend='freq'
                    )
                    
                    # Add components back to DataFrame
                    df = df.merge(
                        pd.DataFrame({
                            f'{col}_trend': decomposition.trend,
                            f'{col}_seasonal': decomposition.seasonal,
                            f'{col}_residual': decomposition.resid
                        }).reset_index(),
                        left_on='date', right_on='date', how='left'
                    )
                    
                    # Calculate seasonal strength
                    df[f'{col}_seasonal_strength'] = (
                        df[f'{col}_seasonal'].abs() / 
                        (df[f'{col}_seasonal'].abs() + df[f'{col}_residual'].abs())
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Decomposition failed for {col}: {e}")
                    continue
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in seasonal decomposition: {e}")
            return df
    
    def detect_change_points(self, df: pd.DataFrame, 
                           columns: List[str],
                           min_size: int = 10) -> pd.DataFrame:
        """
        Detect change points in time series
        
        Args:
            df: DataFrame with time series data
            columns: List of columns to analyze
            min_size: Minimum segment size
            
        Returns:
            DataFrame with change point indicators
        """
        try:
            import ruptures as rpt
            
            df = df.copy()
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                series = df[col].dropna()
                if len(series) < 2 * min_size:
                    continue
                
                try:
                    # Use Pelt algorithm for change point detection
                    algo = rpt.Pelt(model="rbf", min_size=min_size).fit(series.values)
                    change_points = algo.predict(pen=10)
                    
                    # Create change point indicator
                    cp_indicator = pd.Series(0, index=df.index)
                    if change_points[:-1]:  # Exclude the last point (end of series)
                        cp_indices = series.index[change_points[:-1]]
                        cp_indicator.loc[cp_indices] = 1
                    
                    df[f'{col}_change_point'] = cp_indicator
                    
                except Exception as e:
                    self.logger.warning(f"Change point detection failed for {col}: {e}")
                    df[f'{col}_change_point'] = 0
            
            return df
            
        except ImportError:
            self.logger.warning("ruptures package not available, skipping change point detection")
            return df
        except Exception as e:
            self.logger.error(f"Error detecting change points: {e}")
            return df
    
    def calculate_extremes(self, df: pd.DataFrame, 
                          columns: List[str],
                          percentile_threshold: float = 95) -> pd.DataFrame:
        """
        Identify extreme values and events
        
        Args:
            df: DataFrame with time series data
            columns: List of columns to analyze
            percentile_threshold: Percentile threshold for extremes
            
        Returns:
            DataFrame with extreme event indicators
        """
        try:
            df = df.copy()
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                # Calculate percentiles
                p_low = np.percentile(df[col].dropna(), 100 - percentile_threshold)
                p_high = np.percentile(df[col].dropna(), percentile_threshold)
                
                # Identify extremes
                df[f'{col}_extreme_low'] = (df[col] < p_low).astype(int)
                df[f'{col}_extreme_high'] = (df[col] > p_high).astype(int)
                df[f'{col}_extreme'] = (
                    df[f'{col}_extreme_low'] | df[f'{col}_extreme_high']
                ).astype(int)
                
                # Consecutive extreme days
                df[f'{col}_consecutive_extremes'] = (
                    df[f'{col}_extreme'].groupby(
                        (df[f'{col}_extreme'] != df[f'{col}_extreme'].shift()).cumsum()
                    ).cumsum() * df[f'{col}_extreme']
                )
                
                # Extreme event duration
                extreme_events = df[f'{col}_extreme'].astype(bool)
                event_groups = extreme_events.ne(extreme_events.shift()).cumsum()
                event_durations = extreme_events.groupby(event_groups).sum()
                
                df[f'{col}_extreme_duration'] = 0
                for group_id, duration in event_durations.items():
                    if duration > 0:
                        mask = event_groups == group_id
                        df.loc[mask, f'{col}_extreme_duration'] = duration
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating extremes: {e}")
            return df
    
    def water_balance_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze water balance dynamics
        
        Args:
            df: DataFrame with precipitation, ET, and soil moisture data
            
        Returns:
            DataFrame with water balance indicators
        """
        try:
            df = df.copy()
            
            # Calculate water balance components
            if all(col in df.columns for col in ['PRECTOTCORR', 'PET_PT_JPL']):
                df['water_balance_daily'] = df['PRECTOTCORR'] - df['PET_PT_JPL']
                
                # Cumulative water balance
                df['water_balance_cumulative'] = df['water_balance_daily'].cumsum()
                
                # Moving water balance
                df['water_balance_7d'] = df['water_balance_daily'].rolling(7).sum()
                df['water_balance_30d'] = df['water_balance_daily'].rolling(30).sum()
                
                # Water stress periods (negative balance for consecutive days)
                negative_balance = df['water_balance_daily'] < 0
                stress_periods = negative_balance.groupby(
                    (negative_balance != negative_balance.shift()).cumsum()
                ).cumsum() * negative_balance
                
                df['water_stress_duration'] = stress_periods
                
                # Drought index (standardized precipitation-evapotranspiration index)
                df['spei'] = self._calculate_spei(df['water_balance_daily'])
                
                # Water surplus/deficit categories
                df['water_status'] = pd.cut(
                    df['water_balance_7d'],
                    bins=[-np.inf, -20, -10, 10, 20, np.inf],
                    labels=['severe_deficit', 'moderate_deficit', 'normal', 
                           'moderate_surplus', 'severe_surplus']
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in water balance analysis: {e}")
            return df
    
    def _calculate_spei(self, water_balance: pd.Series, window: int = 30) -> pd.Series:
        """Calculate Standardized Precipitation-Evapotranspiration Index"""
        try:
            # Rolling sum
            rolling_wb = water_balance.rolling(window=window).sum()
            
            # Standardize
            spei = (rolling_wb - rolling_wb.mean()) / rolling_wb.std()
            
            return spei
            
        except Exception:
            return pd.Series(np.nan, index=water_balance.index)
    
    def generate_summary_statistics(self, df: pd.DataFrame, 
                                  columns: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Generate comprehensive summary statistics
        
        Args:
            df: DataFrame with time series data
            columns: List of columns to summarize
            
        Returns:
            Dictionary of summary statistics
        """
        try:
            summary = {}
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                
                col_summary = {
                    'count': len(series),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'median': series.median(),
                    'q25': series.quantile(0.25),
                    'q75': series.quantile(0.75),
                    'skewness': stats.skew(series),
                    'kurtosis': stats.kurtosis(series),
                    'trend_slope': self._calculate_trend_slope(series),
                    'cv': series.std() / series.mean() if series.mean() != 0 else np.nan,
                    'anomaly_rate': df.get(f'{col}_anomaly', pd.Series(0)).mean(),
                    'missing_rate': df[col].isna().mean()
                }
                
                # Add stationarity test
                try:
                    adf_result = adfuller(series.values)
                    col_summary['adf_statistic'] = adf_result[0]
                    col_summary['adf_pvalue'] = adf_result[1]
                    col_summary['is_stationary'] = adf_result[1] < 0.05
                except:
                    col_summary['is_stationary'] = None
                
                summary[col] = col_summary
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary statistics: {e}")
            return {}