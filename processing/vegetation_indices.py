"""
Vegetation indices calculation for water stress detection
Includes NDVI, GNDVI, NDWI, CWSI and other stress indicators
"""
import numpy as np
import pandas as pd
import ee
from typing import Dict, List, Tuple, Optional, Any
import logging

class VegetationIndices:
    """Calculate vegetation and water stress indices from satellite data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_ndvi(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index
        
        Args:
            red: Red band reflectance (0.6-0.7 μm)
            nir: Near-infrared reflectance (0.8-1.1 μm)
            
        Returns:
            NDVI values (-1 to 1)
        """
        try:
            return np.where(
                (nir + red) != 0,
                (nir - red) / (nir + red),
                np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating NDVI: {e}")
            return np.full_like(red, np.nan)
    
    def calculate_gndvi(self, green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate Green Normalized Difference Vegetation Index
        
        Args:
            green: Green band reflectance (0.5-0.6 μm)
            nir: Near-infrared reflectance
            
        Returns:
            GNDVI values (-1 to 1)
        """
        try:
            return np.where(
                (nir + green) != 0,
                (nir - green) / (nir + green),
                np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating GNDVI: {e}")
            return np.full_like(green, np.nan)
    
    def calculate_ndwi(self, green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Water Index
        
        Args:
            green: Green band reflectance
            nir: Near-infrared reflectance
            
        Returns:
            NDWI values (-1 to 1)
        """
        try:
            return np.where(
                (green + nir) != 0,
                (green - nir) / (green + nir),
                np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating NDWI: {e}")
            return np.full_like(green, np.nan)
    
    def calculate_ndwi_gao(self, nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """
        Calculate NDWI using Gao's formulation (NIR-SWIR)
        
        Args:
            nir: Near-infrared reflectance
            swir: Short-wave infrared reflectance (1.55-1.75 μm)
            
        Returns:
            NDWI values (-1 to 1)
        """
        try:
            return np.where(
                (nir + swir) != 0,
                (nir - swir) / (nir + swir),
                np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating NDWI (Gao): {e}")
            return np.full_like(nir, np.nan)
    
    def calculate_evi(self, blue: np.ndarray, red: np.ndarray, nir: np.ndarray,
                     G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0) -> np.ndarray:
        """
        Calculate Enhanced Vegetation Index
        
        Args:
            blue: Blue band reflectance
            red: Red band reflectance
            nir: Near-infrared reflectance
            G, C1, C2, L: EVI coefficients
            
        Returns:
            EVI values
        """
        try:
            denominator = nir + C1 * red - C2 * blue + L
            return np.where(
                denominator != 0,
                G * (nir - red) / denominator,
                np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating EVI: {e}")
            return np.full_like(nir, np.nan)
    
    def calculate_savi(self, red: np.ndarray, nir: np.ndarray, L: float = 0.5) -> np.ndarray:
        """
        Calculate Soil Adjusted Vegetation Index
        
        Args:
            red: Red band reflectance
            nir: Near-infrared reflectance
            L: Soil brightness correction factor (0.5 for intermediate vegetation)
            
        Returns:
            SAVI values
        """
        try:
            denominator = nir + red + L
            return np.where(
                denominator != 0,
                (1 + L) * (nir - red) / denominator,
                np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating SAVI: {e}")
            return np.full_like(nir, np.nan)
    
    def calculate_cwsi(self, surface_temp: np.ndarray, air_temp: np.ndarray,
                      vpd: np.ndarray, net_radiation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate Crop Water Stress Index
        
        Args:
            surface_temp: Surface temperature (°C)
            air_temp: Air temperature (°C)
            vpd: Vapor pressure deficit (kPa)
            net_radiation: Net radiation (optional, for improved accuracy)
            
        Returns:
            CWSI values (0-1, where 1 indicates maximum stress)
        """
        try:
            # Temperature difference
            dt = surface_temp - air_temp
            
            # Simplified CWSI calculation without wind speed
            # Upper baseline (non-transpiring crop)
            dt_upper = vpd * 3.0  # Simplified relationship
            
            # Lower baseline (well-watered crop)
            dt_lower = vpd * 0.5 - 2.0  # Simplified relationship
            
            # CWSI calculation
            cwsi = np.where(
                (dt_upper - dt_lower) != 0,
                (dt - dt_lower) / (dt_upper - dt_lower),
                np.nan
            )
            
            # Constrain CWSI between 0 and 1
            cwsi = np.clip(cwsi, 0, 1)
            
            return cwsi
            
        except Exception as e:
            self.logger.error(f"Error calculating CWSI: {e}")
            return np.full_like(surface_temp, np.nan)
    
    def calculate_lai_from_ndvi(self, ndvi: np.ndarray, crop_type: str = 'soybean') -> np.ndarray:
        """
        Estimate Leaf Area Index from NDVI
        
        Args:
            ndvi: NDVI values
            crop_type: Type of crop for specific LAI relationship
            
        Returns:
            LAI estimates
        """
        try:
            # Crop-specific NDVI-LAI relationships (simplified)
            if crop_type.lower() == 'soybean':
                # LAI = a * NDVI / (1 - NDVI) based on soybean studies
                lai = np.where(
                    (ndvi > 0) & (ndvi < 0.95),
                    2.5 * ndvi / (1 - ndvi),
                    np.nan
                )
            elif crop_type.lower() == 'corn':
                lai = np.where(
                    (ndvi > 0) & (ndvi < 0.95),
                    3.0 * ndvi / (1 - ndvi),
                    np.nan
                )
            elif crop_type.lower() == 'wheat':
                lai = np.where(
                    (ndvi > 0) & (ndvi < 0.95),
                    2.0 * ndvi / (1 - ndvi),
                    np.nan
                )
            else:
                # Generic relationship
                lai = np.where(
                    (ndvi > 0) & (ndvi < 0.95),
                    2.5 * ndvi / (1 - ndvi),
                    np.nan
                )
            
            # Constrain LAI to reasonable values (0-8)
            lai = np.clip(lai, 0, 8)
            
            return lai
            
        except Exception as e:
            self.logger.error(f"Error calculating LAI: {e}")
            return np.full_like(ndvi, np.nan)
    
    def calculate_indices_ee(self, image: ee.Image) -> ee.Image:
        """
        Calculate vegetation indices using Google Earth Engine
        
        Args:
            image: Earth Engine image with required bands
            
        Returns:
            Image with added vegetation indices
        """
        try:
            # Extract bands (Sentinel-2 band names)
            blue = image.select('B2')
            green = image.select('B3')
            red = image.select('B4')
            nir = image.select('B8')
            swir1 = image.select('B11')
            swir2 = image.select('B12')
            
            # Calculate NDVI
            ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
            
            # Calculate GNDVI
            gndvi = nir.subtract(green).divide(nir.add(green)).rename('GNDVI')
            
            # Calculate NDWI (McFeeters)
            ndwi_mcfeeters = green.subtract(nir).divide(green.add(nir)).rename('NDWI_McFeeters')
            
            # Calculate NDWI (Gao)
            ndwi_gao = nir.subtract(swir1).divide(nir.add(swir1)).rename('NDWI_Gao')
            
            # Calculate EVI
            evi = image.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                {
                    'NIR': nir,
                    'RED': red,
                    'BLUE': blue
                }
            ).rename('EVI')
            
            # Calculate SAVI
            savi = image.expression(
                '(1 + L) * (NIR - RED) / (NIR + RED + L)',
                {
                    'NIR': nir,
                    'RED': red,
                    'L': 0.5
                }
            ).rename('SAVI')
            
            # Calculate LAI from NDVI
            lai = image.expression(
                '2.5 * NDVI / (1 - NDVI)',
                {'NDVI': ndvi}
            ).where(ndvi.gt(0).And(ndvi.lt(0.95))).rename('LAI')
            
            # Add all indices to the image
            indices_image = image.addBands([
                ndvi, gndvi, ndwi_mcfeeters, ndwi_gao, evi, savi, lai
            ])
            
            return indices_image
            
        except Exception as e:
            self.logger.error(f"Error calculating indices in Earth Engine: {e}")
            return image
    
    def calculate_stress_indicators(self, df: pd.DataFrame, crop_type: str = 'soybean') -> pd.DataFrame:
        """
        Calculate water stress indicators from time series data
        
        Args:
            df: DataFrame with vegetation indices and weather data
            crop_type: Type of crop for thresholds
            
        Returns:
            DataFrame with stress indicators
        """
        try:
            from ..config.crop_parameters import crop_config
            
            # Get crop-specific thresholds
            thresholds = crop_config.get_stress_thresholds(crop_type)
            
            # NDVI-based stress
            if 'NDVI' in df.columns:
                df['NDVI_stress'] = (df['NDVI'] < thresholds.get('ndvi_min', 0.6)).astype(int)
                df['NDVI_decline'] = df['NDVI'].rolling(window=7).mean().diff() < -0.05
            
            # NDWI-based stress
            if 'NDWI_Gao' in df.columns:
                df['NDWI_stress'] = (df['NDWI_Gao'] < thresholds.get('ndwi_stress', 0.3)).astype(int)
            
            # CWSI-based stress
            if 'CWSI' in df.columns:
                df['CWSI_stress'] = (df['CWSI'] > thresholds.get('cwsi_stress', 0.4)).astype(int)
            
            # Combined stress index
            stress_columns = [col for col in df.columns if col.endswith('_stress')]
            if stress_columns:
                df['combined_stress_index'] = df[stress_columns].sum(axis=1) / len(stress_columns)
            
            # Vegetation vigor (combination of indices)
            if all(col in df.columns for col in ['NDVI', 'EVI', 'SAVI']):
                df['vegetation_vigor'] = (df['NDVI'] + df['EVI'] + df['SAVI']) / 3
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating stress indicators: {e}")
            return df
    
    def calculate_temporal_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate temporal derivatives and trends of vegetation indices
        
        Args:
            df: DataFrame with time series vegetation indices
            
        Returns:
            DataFrame with temporal derivatives
        """
        try:
            # Ensure data is sorted by date
            df = df.sort_values('date').copy()
            
            # Calculate rolling means for smoothing
            for col in ['NDVI', 'GNDVI', 'NDWI_Gao', 'EVI', 'SAVI']:
                if col in df.columns:
                    # 7-day rolling mean
                    df[f'{col}_7d_mean'] = df[col].rolling(window=7, center=True).mean()
                    
                    # Rate of change
                    df[f'{col}_rate_change'] = df[f'{col}_7d_mean'].diff()
                    
                    # Seasonal trend (30-day)
                    df[f'{col}_30d_trend'] = df[col].rolling(window=30).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 30 else np.nan
                    )
            
            # Calculate vegetation phenology metrics
            if 'NDVI' in df.columns:
                # Growing season start (NDVI > 0.3 for 3 consecutive days)
                ndvi_threshold = df['NDVI'] > 0.3
                df['growing_season'] = ndvi_threshold.rolling(window=3).sum() >= 3
                
                # Peak vegetation (local maxima)
                df['NDVI_peak'] = (
                    (df['NDVI'] > df['NDVI'].shift(1)) & 
                    (df['NDVI'] > df['NDVI'].shift(-1))
                )
                
                # Senescence detection (sustained decline)
                df['senescence'] = (
                    df['NDVI_rate_change'].rolling(window=5).mean() < -0.01
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal derivatives: {e}")
            return df
    
    def calculate_anomalies(self, df: pd.DataFrame, baseline_years: int = 3) -> pd.DataFrame:
        """
        Calculate vegetation index anomalies compared to historical baseline
        
        Args:
            df: DataFrame with vegetation indices
            baseline_years: Number of years for baseline calculation
            
        Returns:
            DataFrame with anomaly indicators
        """
        try:
            # Group by day of year for climatology
            df['day_of_year'] = df['date'].dt.dayofyear
            
            # Calculate historical means and std for each day of year
            for col in ['NDVI', 'GNDVI', 'NDWI_Gao', 'EVI']:
                if col in df.columns:
                    # Calculate rolling climatology (±15 days window)
                    climatology_stats = []
                    
                    for doy in range(1, 367):
                        # Get days within ±15 days window (circular)
                        doy_range = [(doy + i - 15) % 366 + 1 for i in range(31)]
                        
                        historical_data = df[
                            df['day_of_year'].isin(doy_range) & 
                            (df['date'] < df['date'].max() - pd.Timedelta(days=365))
                        ][col].dropna()
                        
                        if len(historical_data) > 5:  # Minimum data points
                            mean_val = historical_data.mean()
                            std_val = historical_data.std()
                            climatology_stats.append({
                                'day_of_year': doy,
                                f'{col}_clim_mean': mean_val,
                                f'{col}_clim_std': std_val
                            })
                    
                    if climatology_stats:
                        clim_df = pd.DataFrame(climatology_stats)
                        df = df.merge(clim_df, on='day_of_year', how='left')
                        
                        # Calculate anomalies
                        df[f'{col}_anomaly'] = df[col] - df[f'{col}_clim_mean']
                        df[f'{col}_anomaly_std'] = (
                            df[f'{col}_anomaly'] / df[f'{col}_clim_std']
                        )
                        
                        # Flag significant anomalies (> 2 standard deviations)
                        df[f'{col}_severe_anomaly'] = (
                            np.abs(df[f'{col}_anomaly_std']) > 2
                        ).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating anomalies: {e}")
            return df