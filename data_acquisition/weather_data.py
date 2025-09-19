"""
Weather data acquisition using NASA POWER API
Provides meteorological data for water stress analysis
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import asyncio
import aiohttp
from config.api_config import api_config

class WeatherDataAcquisition:
    """Handles weather data acquisition from NASA POWER API"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = api_config.nasa_power['base_url']
        self.parameters = api_config.nasa_power['parameters']
        self.community = api_config.nasa_power['community']

    async def get_weather_data_async(self,
                                   coordinates: List[Tuple[float, float]],
                                   start_date: str,
                                   end_date: str,
                                   parameters: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Asynchronously retrieve weather data for multiple locations

        Args:
            coordinates: List of (longitude, latitude) tuples
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            parameters: List of weather parameters to retrieve

        Returns:
            Dictionary mapping location to weather DataFrame
        """
        if parameters is None:
            parameters = self.parameters

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, (lon, lat) in enumerate(coordinates):
                task = self._fetch_location_data_async(
                    session, lon, lat, start_date, end_date, parameters, f"location_{i}"
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            weather_data = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error fetching data for location {i}: {result}")
                    continue

                location_key = f"location_{i}_lon_{coordinates[i][0]}_lat_{coordinates[i][1]}"
                weather_data[location_key] = result

            return weather_data

    async def _fetch_location_data_async(self,
                                       session: aiohttp.ClientSession,
                                       longitude: float,
                                       latitude: float,
                                       start_date: str,
                                       end_date: str,
                                       parameters: List[str],
                                       location_id: str) -> pd.DataFrame:
        """Fetch weather data for a specific location asynchronously"""
        try:
            params = {
                'parameters': ','.join(parameters),
                'community': self.community,
                'longitude': longitude,
                'latitude': latitude,
                'start': start_date.replace('-', ''),
                'end': end_date.replace('-', ''),
                'format': 'JSON'
            }

            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    df = self._parse_nasa_power_response(data, location_id)
                    return df
                else:
                    raise Exception(f"API request failed with status {response.status}")

        except Exception as e:
            self.logger.error(f"Error fetching data for {location_id}: {e}")
            raise

    def get_weather_data(self,
                        longitude: float,
                        latitude: float,
                        start_date: str,
                        end_date: str,
                        parameters: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retrieve weather data for a single location (synchronous)

        Args:
            longitude: Longitude coordinate
            latitude: Latitude coordinate
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            parameters: List of weather parameters to retrieve

        Returns:
            DataFrame containing weather data
        """
        if parameters is None:
            parameters = self.parameters

        try:
            params = {
                'parameters': ','.join(parameters),
                'community': self.community,
                'longitude': longitude,
                'latitude': latitude,
                'start': start_date.replace('-', ''),
                'end': end_date.replace('-', ''),
                'format': 'JSON'
            }

            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            df = self._parse_nasa_power_response(data, f"lon_{longitude}_lat_{latitude}")

            return df

        except Exception as e:
            self.logger.error(f"Error retrieving weather data: {e}")
            raise

    def _parse_nasa_power_response(self, data: Dict, location_id: str) -> pd.DataFrame:
        """Parse NASA POWER API response into pandas DataFrame"""
        try:
            properties = data.get('properties', {})
            parameter_data = properties.get('parameter', {})

            if not parameter_data:
                return pd.DataFrame()

            # Extract dates and create DataFrame
            dates = []
            records = []

            # Get the first parameter to extract dates
            first_param = list(parameter_data.keys())[0]
            date_keys = list(parameter_data[first_param].keys())

            for date_key in date_keys:
                try:
                    date = datetime.strptime(date_key, '%Y%m%d')
                    dates.append(date)

                    record = {'date': date, 'location_id': location_id}

                    for param, values in parameter_data.items():
                        value = values.get(date_key)
                        if value != -999.0:  # NASA POWER missing value flag
                            record[param] = value
                        else:
                            record[param] = np.nan

                    records.append(record)

                except ValueError:
                    continue

            df = pd.DataFrame(records)

            if not df.empty:
                df = df.sort_values('date').reset_index(drop=True)
                df = self._add_derived_parameters(df)

            return df

        except Exception as e:
            self.logger.error(f"Error parsing NASA POWER response: {e}")
            return pd.DataFrame()

    def _add_derived_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived meteorological parameters"""
        try:
            # Vapor Pressure Deficit (VPD)
            if 'T2M' in df.columns and 'T2M_MAX' in df.columns and 'T2M_MIN' in df.columns:
                df['VPD'] = self._calculate_vpd(
                    df['T2M_MAX'], df['T2M_MIN'], df['T2M']
                )

            # Growing Degree Days (GDD) - base temperature 10°C
            if 'T2M_MAX' in df.columns and 'T2M_MIN' in df.columns:
                df['GDD'] = self._calculate_gdd(df['T2M_MAX'], df['T2M_MIN'], base_temp=10)

            # Precipitation cumulative
            if 'PRECTOTCORR' in df.columns:
                df['PRECIP_CUMULATIVE_7D'] = df['PRECTOTCORR'].rolling(window=7).sum()
                df['PRECIP_CUMULATIVE_30D'] = df['PRECTOTCORR'].rolling(window=30).sum()

            # Temperature stress indicators
            if 'T2M_MAX' in df.columns:
                df['HEAT_STRESS_DAYS'] = (df['T2M_MAX'] > 32).astype(int)  # Days above 32°C

            # Evapotranspiration water balance
            if 'EVPTRNS' in df.columns and 'PRECTOTCORR' in df.columns:
                df['WATER_BALANCE'] = df['PRECTOTCORR'] - df['EVPTRNS']
                df['WATER_BALANCE_CUMULATIVE'] = df['WATER_BALANCE'].cumsum()

            return df

        except Exception as e:
            self.logger.error(f"Error adding derived parameters: {e}")
            return df

    def _calculate_vpd(self, tmax: pd.Series, tmin: pd.Series, tmean: pd.Series) -> pd.Series:
        """Calculate Vapor Pressure Deficit"""
        try:
            # Saturation vapor pressure at mean temperature (kPa)
            es_mean = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))

            # Saturation vapor pressure at max and min temperatures
            es_max = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
            es_min = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))

            # Mean saturation vapor pressure
            es = (es_max + es_min) / 2

            # Actual vapor pressure (assuming minimum temperature = dew point)
            ea = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))

            # VPD
            vpd = es - ea

            return vpd

        except Exception:
            return pd.Series(np.nan, index=tmean.index)

    def _calculate_gdd(self, tmax: pd.Series, tmin: pd.Series, base_temp: float = 10) -> pd.Series:
        """Calculate Growing Degree Days"""
        try:
            tmean = (tmax + tmin) / 2
            gdd = np.maximum(tmean - base_temp, 0)
            return gdd

        except Exception:
            return pd.Series(np.nan, index=tmax.index)

    def get_weather_summary(self, df: pd.DataFrame, period_days: int = 30) -> Dict[str, float]:
        """Generate weather summary statistics for the last period"""
        try:
            if df.empty:
                return {}

            # Get recent data
            recent_data = df.tail(period_days)

            summary = {
                'mean_temperature': recent_data['T2M'].mean() if 'T2M' in recent_data.columns else np.nan,
                'max_temperature': recent_data['T2M_MAX'].max() if 'T2M_MAX' in recent_data.columns else np.nan,
                'min_temperature': recent_data['T2M_MIN'].min() if 'T2M_MIN' in recent_data.columns else np.nan,
                'total_precipitation': recent_data['PRECTOTCORR'].sum() if 'PRECTOTCORR' in recent_data.columns else np.nan,
                'mean_evapotranspiration': recent_data['EVPTRNS'].mean() if 'EVPTRNS' in recent_data.columns else np.nan,
                'mean_vpd': recent_data['VPD'].mean() if 'VPD' in recent_data.columns else np.nan,
                'gdd_sum': recent_data['GDD'].sum() if 'GDD' in recent_data.columns else np.nan,
                'heat_stress_days': recent_data['HEAT_STRESS_DAYS'].sum() if 'HEAT_STRESS_DAYS' in recent_data.columns else np.nan,
                'water_balance': recent_data['WATER_BALANCE'].sum() if 'WATER_BALANCE' in recent_data.columns else np.nan
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error generating weather summary: {e}")
            return {}

    def get_historical_weather(self,
                              longitude: float,
                              latitude: float,
                              years_back: int = 3) -> pd.DataFrame:
        """Get historical weather data for model training"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years_back)

            df = self.get_weather_data(
                longitude,
                latitude,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            return df

        except Exception as e:
            self.logger.error(f"Error retrieving historical weather data: {e}")
            raise
