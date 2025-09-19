"""
Soil data acquisition using SoilGrids API
Provides soil properties for water stress analysis
"""
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import asyncio
import aiohttp
from config.api_config import api_config

class SoilDataAcquisition:
    """Handles soil data acquisition from SoilGrids API"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = api_config.soilgrids['base_url']
        self.properties = api_config.soilgrids['properties']
        self.depths = api_config.soilgrids['depths']

    async def get_soil_data_async(self,
                                coordinates: List[Tuple[float, float]],
                                properties: Optional[List[str]] = None,
                                depths: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Asynchronously retrieve soil data for multiple locations

        Args:
            coordinates: List of (longitude, latitude) tuples
            properties: List of soil properties to retrieve
            depths: List of soil depths to retrieve

        Returns:
            Dictionary mapping location to soil DataFrame
        """
        if properties is None:
            properties = self.properties
        if depths is None:
            depths = self.depths

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, (lon, lat) in enumerate(coordinates):
                task = self._fetch_soil_location_async(
                    session, lon, lat, properties, depths, f"location_{i}"
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            soil_data = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error fetching soil data for location {i}: {result}")
                    continue

                location_key = f"location_{i}_lon_{coordinates[i][0]}_lat_{coordinates[i][1]}"
                soil_data[location_key] = result

            return soil_data

    async def _fetch_soil_location_async(self,
                                       session: aiohttp.ClientSession,
                                       longitude: float,
                                       latitude: float,
                                       properties: List[str],
                                       depths: List[str],
                                       location_id: str) -> pd.DataFrame:
        """Fetch soil data for a specific location asynchronously"""
        try:
            params = {
                'lon': longitude,
                'lat': latitude,
                'property': properties,
                'depth': depths,
                'value': 'mean'  # Get mean values
            }

            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    df = self._parse_soilgrids_response(data, location_id)
                    return df
                else:
                    raise Exception(f"API request failed with status {response.status}")

        except Exception as e:
            self.logger.error(f"Error fetching soil data for {location_id}: {e}")
            raise

    def get_soil_data(self,
                     longitude: float,
                     latitude: float,
                     properties: Optional[List[str]] = None,
                     depths: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retrieve soil data for a single location (synchronous)

        Args:
            longitude: Longitude coordinate
            latitude: Latitude coordinate
            properties: List of soil properties to retrieve
            depths: List of soil depths to retrieve

        Returns:
            DataFrame containing soil data
        """
        if properties is None:
            properties = self.properties
        if depths is None:
            depths = self.depths

        try:
            params = {
                'lon': longitude,
                'lat': latitude,
                'property': properties,
                'depth': depths,
                'value': 'mean'
            }

            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            df = self._parse_soilgrids_response(data, f"lon_{longitude}_lat_{latitude}")

            return df

        except Exception as e:
            self.logger.error(f"Error retrieving soil data: {e}")
            raise

    def _parse_soilgrids_response(self, data: Dict, location_id: str) -> pd.DataFrame:
        """Parse SoilGrids API response into pandas DataFrame"""
        try:
            properties_data = data.get('properties', {})
            layers = properties_data.get('layers', [])

            if not layers:
                return pd.DataFrame()

            records = []

            for layer in layers:
                layer_name = layer.get('name', '')
                layer_unit = layer.get('unit_measure', {}).get('mapped_units', '')
                depths_data = layer.get('depths', [])

                for depth_info in depths_data:
                    depth_label = depth_info.get('label', '')
                    depth_range = depth_info.get('range', {})
                    depth_top = depth_range.get('top_depth', 0)
                    depth_bottom = depth_range.get('bottom_depth', 0)

                    values = depth_info.get('values', {})
                    mean_value = values.get('mean')

                    if mean_value is not None:
                        record = {
                            'location_id': location_id,
                            'property': layer_name,
                            'depth_label': depth_label,
                            'depth_top_cm': depth_top,
                            'depth_bottom_cm': depth_bottom,
                            'value': mean_value,
                            'unit': layer_unit
                        }
                        records.append(record)

            df = pd.DataFrame(records)

            if not df.empty:
                df = self._add_derived_soil_properties(df)
                df = self._pivot_soil_data(df)

            return df

        except Exception as e:
            self.logger.error(f"Error parsing SoilGrids response: {e}")
            return pd.DataFrame()

    def _add_derived_soil_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived soil properties for water stress analysis"""
        try:
            # Add soil texture class based on sand, silt, clay percentages
            if all(prop in df['property'].values for prop in ['sand', 'silt', 'clay']):
                df = self._add_soil_texture_class(df)

            # Calculate available water content (AWC) if bulk density and texture are available
            if 'bdod' in df['property'].values:
                df = self._calculate_awc(df)

            # Add water retention properties
            df = self._add_water_retention_properties(df)

            return df

        except Exception as e:
            self.logger.error(f"Error adding derived soil properties: {e}")
            return df

    def _add_soil_texture_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add USDA soil texture classification"""
        try:
            # This is a simplified version - for production, use the full USDA triangle
            texture_records = []

            locations = df['location_id'].unique()
            depths = df['depth_label'].unique()

            for location in locations:
                for depth in depths:
                    depth_data = df[(df['location_id'] == location) &
                                   (df['depth_label'] == depth)]

                    sand = depth_data[depth_data['property'] == 'sand']['value'].iloc[0] if not depth_data[depth_data['property'] == 'sand'].empty else 0
                    silt = depth_data[depth_data['property'] == 'silt']['value'].iloc[0] if not depth_data[depth_data['property'] == 'silt'].empty else 0
                    clay = depth_data[depth_data['property'] == 'clay']['value'].iloc[0] if not depth_data[depth_data['property'] == 'clay'].empty else 0

                    # Convert from g/kg to percentage if needed
                    if sand > 100:  # Assuming g/kg
                        sand, silt, clay = sand/10, silt/10, clay/10

                    texture_class = self._classify_soil_texture(sand, silt, clay)

                    texture_record = {
                        'location_id': location,
                        'property': 'texture_class',
                        'depth_label': depth,
                        'depth_top_cm': depth_data['depth_top_cm'].iloc[0],
                        'depth_bottom_cm': depth_data['depth_bottom_cm'].iloc[0],
                        'value': texture_class,
                        'unit': 'class'
                    }
                    texture_records.append(texture_record)

            if texture_records:
                texture_df = pd.DataFrame(texture_records)
                df = pd.concat([df, texture_df], ignore_index=True)

            return df

        except Exception as e:
            self.logger.error(f"Error adding soil texture class: {e}")
            return df

    def _classify_soil_texture(self, sand: float, silt: float, clay: float) -> str:
        """Classify soil texture using simplified USDA triangle"""
        try:
            if clay >= 40:
                return 'Clay'
            elif clay >= 30:
                if silt >= 40:
                    return 'Silty Clay'
                else:
                    return 'Sandy Clay'
            elif clay >= 20:
                if sand >= 45:
                    return 'Sandy Clay Loam'
                elif silt >= 50:
                    return 'Silty Clay Loam'
                else:
                    return 'Clay Loam'
            elif silt >= 80:
                return 'Silt'
            elif silt >= 50:
                return 'Silt Loam'
            elif sand >= 85:
                return 'Sand'
            elif sand >= 70:
                return 'Loamy Sand'
            else:
                return 'Loam'

        except Exception:
            return 'Unknown'

    def _calculate_awc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Available Water Content based on texture and bulk density"""
        try:
            awc_records = []

            locations = df['location_id'].unique()
            depths = df['depth_label'].unique()

            for location in locations:
                for depth in depths:
                    depth_data = df[(df['location_id'] == location) &
                                   (df['depth_label'] == depth)]

                    # Get bulk density (kg/m³)
                    bdod_data = depth_data[depth_data['property'] == 'bdod']
                    if bdod_data.empty:
                        continue

                    bulk_density = bdod_data['value'].iloc[0] / 100  # Convert to g/cm³

                    # Get texture components
                    sand = depth_data[depth_data['property'] == 'sand']['value'].iloc[0] if not depth_data[depth_data['property'] == 'sand'].empty else 0
                    clay = depth_data[depth_data['property'] == 'clay']['value'].iloc[0] if not depth_data[depth_data['property'] == 'clay'].empty else 0

                    if sand > 100:  # Convert from g/kg to %
                        sand, clay = sand/10, clay/10

                    # Simplified AWC calculation (cm³/cm³)
                    # This is a simplified version - for production use pedotransfer functions
                    awc = self._estimate_awc_from_texture(sand, clay, bulk_density)

                    awc_record = {
                        'location_id': location,
                        'property': 'awc',
                        'depth_label': depth,
                        'depth_top_cm': depth_data['depth_top_cm'].iloc[0],
                        'depth_bottom_cm': depth_data['depth_bottom_cm'].iloc[0],
                        'value': awc,
                        'unit': 'cm3/cm3'
                    }
                    awc_records.append(awc_record)

            if awc_records:
                awc_df = pd.DataFrame(awc_records)
                df = pd.concat([df, awc_df], ignore_index=True)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating AWC: {e}")
            return df

    def _estimate_awc_from_texture(self, sand: float, clay: float, bulk_density: float) -> float:
        """Estimate AWC using simplified pedotransfer function"""
        try:
            # Simplified Saxton-Rawls equations
            silt = 100 - sand - clay

            # Field capacity (θ33)
            fc = 0.2576 - 0.0020*sand + 0.0036*clay + 0.0299*np.log10(clay + 1)
            fc = fc + (1.283 * fc**2 - 0.374 * fc - 0.015)

            # Wilting point (θ1500)
            wp = 0.0260 + 0.0050*clay + 0.0158*np.log10(clay + 1)
            wp = wp + (0.002 * clay**2 - 0.044 * wp - 0.002)

            # Available water content
            awc = fc - wp

            # Adjust for bulk density effect
            awc = awc * (1.5 - bulk_density)

            return max(0, min(0.5, awc))  # Constrain between 0 and 0.5

        except Exception:
            return 0.15  # Default moderate AWC

    def _add_water_retention_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add water retention curve parameters"""
        try:
            # Add hydraulic conductivity estimates based on texture
            ksat_records = []

            locations = df['location_id'].unique()
            depths = df['depth_label'].unique()

            for location in locations:
                for depth in depths:
                    depth_data = df[(df['location_id'] == location) &
                                   (df['depth_label'] == depth)]

                    # Get texture for Ksat estimation
                    sand = depth_data[depth_data['property'] == 'sand']['value'].iloc[0] if not depth_data[depth_data['property'] == 'sand'].empty else 0
                    clay = depth_data[depth_data['property'] == 'clay']['value'].iloc[0] if not depth_data[depth_data['property'] == 'clay'].empty else 0

                    if sand > 100:  # Convert from g/kg to %
                        sand, clay = sand/10, clay/10

                    # Estimate saturated hydraulic conductivity (cm/day)
                    ksat = self._estimate_ksat(sand, clay)

                    ksat_record = {
                        'location_id': location,
                        'property': 'ksat',
                        'depth_label': depth,
                        'depth_top_cm': depth_data['depth_top_cm'].iloc[0],
                        'depth_bottom_cm': depth_data['depth_bottom_cm'].iloc[0],
                        'value': ksat,
                        'unit': 'cm/day'
                    }
                    ksat_records.append(ksat_record)

            if ksat_records:
                ksat_df = pd.DataFrame(ksat_records)
                df = pd.concat([df, ksat_df], ignore_index=True)

            return df

        except Exception as e:
            self.logger.error(f"Error adding water retention properties: {e}")
            return df

    def _estimate_ksat(self, sand: float, clay: float) -> float:
        """Estimate saturated hydraulic conductivity"""
        try:
            # Simplified Cosby et al. (1984) equation
            # Log10(Ksat) = -0.6 + 0.0126*sand - 0.0064*clay
            log_ksat = -0.6 + 0.0126*sand - 0.0064*clay
            ksat_cm_hr = 10**log_ksat  # cm/hr
            ksat_cm_day = ksat_cm_hr * 24  # cm/day

            return max(0.1, min(100, ksat_cm_day))  # Constrain reasonable values

        except Exception:
            return 10.0  # Default moderate permeability

    def _pivot_soil_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot soil data for easier analysis"""
        try:
            if df.empty:
                return df

            # Create a pivot table with properties as columns
            pivot_df = df.pivot_table(
                index=['location_id', 'depth_label', 'depth_top_cm', 'depth_bottom_cm'],
                columns='property',
                values='value',
                aggfunc='first'
            ).reset_index()

            # Flatten column names
            pivot_df.columns.name = None

            return pivot_df

        except Exception as e:
            self.logger.error(f"Error pivoting soil data: {e}")
            return df

    def get_soil_summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate soil summary for the root zone (0-100cm)"""
        try:
            if df.empty:
                return {}

            # Filter for root zone depths
            root_zone = df[(df['depth_top_cm'] >= 0) & (df['depth_bottom_cm'] <= 100)]

            if root_zone.empty:
                return {}

            summary = {}

            # Calculate weighted averages by depth
            for prop in ['sand', 'silt', 'clay', 'bdod', 'awc', 'ksat']:
                if prop in root_zone.columns:
                    prop_data = root_zone[root_zone[prop].notna()]
                    if not prop_data.empty:
                        # Weight by depth interval
                        depths = prop_data['depth_bottom_cm'] - prop_data['depth_top_cm']
                        weighted_avg = (prop_data[prop] * depths).sum() / depths.sum()
                        summary[f'{prop}_weighted_avg'] = weighted_avg

            # Add texture class for surface layer (0-30cm)
            surface = root_zone[root_zone['depth_bottom_cm'] <= 30]
            if not surface.empty and 'texture_class' in surface.columns:
                texture_classes = surface['texture_class'].dropna()
                if not texture_classes.empty:
                    summary['surface_texture'] = texture_classes.iloc[0]

            return summary

        except Exception as e:
            self.logger.error(f"Error generating soil summary: {e}")
            return {}
