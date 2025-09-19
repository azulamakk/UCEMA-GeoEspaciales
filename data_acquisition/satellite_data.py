"""
Satellite data acquisition using Google Earth Engine API
Supports Sentinel-2 and Landsat data for vegetation and water stress indices
"""
import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from config.api_config import api_config

class SatelliteDataAcquisition:
    """Handles satellite data acquisition from Google Earth Engine"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_earth_engine()

    def _initialize_earth_engine(self):
        """Initialize Google Earth Engine authentication"""
        try:
            if api_config.google_earth_engine['use_service_account']:
                service_account_key = api_config.google_earth_engine['service_account_key']
                credentials = ee.ServiceAccountCredentials(
                    email=None,
                    key_file=service_account_key
                )
                ee.Initialize(credentials)
            else:
                ee.Authenticate()
                ee.Initialize()

            self.logger.info("Google Earth Engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Earth Engine: {e}")
            raise

    def get_sentinel2_data(self,
                          geometry: List[List[float]],
                          start_date: str,
                          end_date: str,
                          cloud_cover_threshold: float = 20.0) -> Dict[str, Any]:
        """
        Retrieve Sentinel-2 data for specified area and time period

        Args:
            geometry: Polygon coordinates [[lon, lat], ...]
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            cloud_cover_threshold: Maximum cloud cover percentage

        Returns:
            Dictionary containing processed Sentinel-2 data
        """
        try:
            # Define area of interest
            aoi = ee.Geometry.Polygon(geometry)

            # Get Sentinel-2 Surface Reflectance collection
            s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                           .filterBounds(aoi)
                           .filterDate(start_date, end_date)
                           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_threshold)))

            # Apply cloud masking
            s2_collection = s2_collection.map(self._mask_s2_clouds)

            # Calculate median composite
            s2_median = s2_collection.median().clip(aoi)

            # Get time series data
            time_series = self._extract_time_series(s2_collection, aoi)

            return {
                'collection': s2_collection,
                'median_composite': s2_median,
                'time_series': time_series,
                'geometry': aoi,
                'date_range': (start_date, end_date),
                'image_count': s2_collection.size().getInfo()
            }

        except Exception as e:
            self.logger.error(f"Error retrieving Sentinel-2 data: {e}")
            raise

    def get_landsat_thermal_data(self,
                                geometry: List[List[float]],
                                start_date: str,
                                end_date: str) -> Dict[str, Any]:
        """
        Retrieve Landsat thermal data for CWSI calculation

        Args:
            geometry: Polygon coordinates
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            Dictionary containing Landsat thermal data
        """
        try:
            aoi = ee.Geometry.Polygon(geometry)

            # Get Landsat 8/9 Collection 2 Surface Temperature
            landsat_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                                .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
                                .filterBounds(aoi)
                                .filterDate(start_date, end_date)
                                .filter(ee.Filter.lt('CLOUD_COVER', 30)))

            # Apply cloud masking and scaling
            landsat_collection = landsat_collection.map(self._mask_landsat_clouds)

            # Calculate median composite
            landsat_median = landsat_collection.median().clip(aoi)

            # Get time series for thermal bands
            thermal_time_series = self._extract_thermal_time_series(landsat_collection, aoi)

            return {
                'collection': landsat_collection,
                'median_composite': landsat_median,
                'thermal_time_series': thermal_time_series,
                'geometry': aoi,
                'date_range': (start_date, end_date)
            }

        except Exception as e:
            self.logger.error(f"Error retrieving Landsat thermal data: {e}")
            raise

    def _mask_s2_clouds(self, image):
        """Apply cloud mask to Sentinel-2 images"""
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

        # Scale optical bands
        optical_bands = image.select('B.*').multiply(0.0001)
        return image.addBands(optical_bands, None, True).updateMask(mask)

    def _mask_landsat_clouds(self, image):
        """Apply cloud mask to Landsat images"""
        qa_pixel = image.select('QA_PIXEL')
        cloud_mask = qa_pixel.bitwiseAnd(1 << 3).eq(0)  # Cloud bit
        cloud_shadow_mask = qa_pixel.bitwiseAnd(1 << 4).eq(0)  # Cloud shadow bit

        # Apply scaling factors
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_band = image.select('ST_B.*').multiply(0.00341802).add(149.0)  # Convert to Celsius

        return (image.addBands(optical_bands, None, True)
                    .addBands(thermal_band, None, True)
                    .updateMask(cloud_mask.And(cloud_shadow_mask)))

    def _extract_time_series(self, collection, geometry) -> pd.DataFrame:
        """Extract time series data from image collection with optimization for large areas"""
        try:
            # Calculate area to determine appropriate scale
            area_km2 = geometry.area().divide(1e6).getInfo()
            self.logger.info(f"Processing area: {area_km2:.2f} km²")
            
            # Optimize scale and maxPixels based on area size
            if area_km2 > 100000:  # Very large area (>100,000 km²)
                scale = 1000
                maxPixels = 1e8
                self.logger.info("Using large area optimization (1km scale)")
            elif area_km2 > 10000:  # Large area (>10,000 km²) 
                scale = 500
                maxPixels = 5e8
                self.logger.info("Using medium area optimization (500m scale)")
            else:  # Small-medium area
                scale = 100
                maxPixels = 1e9
                self.logger.info("Using standard resolution (100m scale)")

            # Define function to extract values for each image
            def extract_values(image):
                # Calculate mean values for the geometry
                stats = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=scale,
                    maxPixels=maxPixels,
                    bestEffort=True
                )

                # Get image date
                date = image.get('system:time_start')

                return ee.Feature(None, stats.set('date', date))

            # Map over collection and extract to DataFrame
            time_series_fc = collection.map(extract_values)
            
            # Add timeout handling for getInfo()
            self.logger.info("Extracting time series data from Google Earth Engine...")
            try:
                time_series_list = time_series_fc.getInfo()['features']
            except Exception as e:
                if "timed out" in str(e).lower():
                    self.logger.warning("GEE timeout detected, attempting with smaller sample...")
                    # Reduce collection size by taking every other image
                    reduced_collection = collection.filter(ee.Filter.calendarRange(1, 31, 'day_of_month').eq(ee.Filter.calendarRange(1, 15, 'day_of_month')))
                    time_series_fc = reduced_collection.map(extract_values)
                    time_series_list = time_series_fc.getInfo()['features']
                else:
                    raise e

            # Convert to pandas DataFrame
            data = []
            for feature in time_series_list:
                props = feature['properties']
                if 'date' in props:
                    date = datetime.fromtimestamp(props['date'] / 1000)
                    row = {'date': date}
                    row.update({k: v for k, v in props.items() if k != 'date' and v is not None})
                    data.append(row)

            df = pd.DataFrame(data)
            if not df.empty:
                df = df.sort_values('date').reset_index(drop=True)

            return df

        except Exception as e:
            self.logger.error(f"Error extracting time series: {e}")
            return pd.DataFrame()

    def _extract_thermal_time_series(self, collection, geometry) -> pd.DataFrame:
        """Extract thermal time series from Landsat collection"""
        try:
            def extract_thermal_values(image):
                stats = image.select(['ST_B10', 'SR_B4', 'SR_B5']).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=100,  # Use 100m resolution to reduce pixel count
                    maxPixels=1e10,  # Increase maxPixels limit
                    bestEffort=True  # Use best effort to aggregate at suitable scale
                )

                date = image.get('system:time_start')
                return ee.Feature(None, stats.set('date', date))

            thermal_fc = collection.map(extract_thermal_values)
            thermal_list = thermal_fc.getInfo()['features']

            data = []
            for feature in thermal_list:
                props = feature['properties']
                if 'date' in props:
                    date = datetime.fromtimestamp(props['date'] / 1000)
                    row = {'date': date}
                    row.update({k: v for k, v in props.items() if k != 'date' and v is not None})
                    data.append(row)

            df = pd.DataFrame(data)
            if not df.empty:
                df = df.sort_values('date').reset_index(drop=True)

            return df

        except Exception as e:
            self.logger.error(f"Error extracting thermal time series: {e}")
            return pd.DataFrame()

    def get_historical_data(self,
                           geometry: List[List[float]],
                           years_back: int = 3,
                           crop_type: str = 'soybean') -> Dict[str, Any]:
        """
        Get historical satellite data for model training

        Args:
            geometry: Area of interest coordinates
            years_back: Number of years of historical data to retrieve
            crop_type: Type of crop for seasonal filtering

        Returns:
            Dictionary containing historical data
        """
        try:
            from ..config.crop_parameters import crop_config

            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years_back)

            # Get crop growing season months
            crop_params = crop_config.get_crop_parameters(crop_type)
            if not crop_params:
                raise ValueError(f"Unknown crop type: {crop_type}")

            # Get data for each growing season
            historical_data = {}

            for year in range(years_back):
                season_start = datetime(end_date.year - year, crop_params.growing_season_months[0], 1)
                season_end = datetime(end_date.year - year + 1, crop_params.growing_season_months[1], 28)

                # Handle season crossing year boundary
                if crop_params.growing_season_months[0] > crop_params.growing_season_months[1]:
                    season_end = datetime(end_date.year - year + 1, crop_params.growing_season_months[1], 28)

                season_key = f"season_{end_date.year - year}"

                # Get Sentinel-2 data
                s2_data = self.get_sentinel2_data(
                    geometry,
                    season_start.strftime('%Y-%m-%d'),
                    season_end.strftime('%Y-%m-%d')
                )

                # Get Landsat thermal data
                thermal_data = self.get_landsat_thermal_data(
                    geometry,
                    season_start.strftime('%Y-%m-%d'),
                    season_end.strftime('%Y-%m-%d')
                )

                historical_data[season_key] = {
                    'sentinel2': s2_data,
                    'landsat_thermal': thermal_data,
                    'season_dates': (season_start, season_end)
                }

            return historical_data

        except Exception as e:
            self.logger.error(f"Error retrieving historical data: {e}")
            raise

    def export_to_drive(self, image, geometry, filename: str, scale: int = 10):
        """Export processed image to Google Drive"""
        try:
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=filename,
                folder='water_stress_detection',
                region=geometry,
                scale=scale,
                crs='EPSG:4326',
                maxPixels=1e9
            )
            task.start()

            self.logger.info(f"Export task started: {filename}")
            return task

        except Exception as e:
            self.logger.error(f"Error exporting to Drive: {e}")
            raise
