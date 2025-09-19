"""
Debug test to identify which data acquisition step is causing issues
"""
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.api_config import api_config
from data_acquisition.satellite_data import SatelliteDataAcquisition
from data_acquisition.weather_data import WeatherDataAcquisition
from data_acquisition.soil_data import SoilDataAcquisition

def test_individual_apis():
    """Test each API individually with a small area"""
    
    # Use a much smaller test area (around Buenos Aires)
    small_geometry = [[-58.5, -34.7], [-58.3, -34.7], [-58.3, -34.5], [-58.5, -34.5], [-58.5, -34.7]]
    
    # Use safe historical dates
    start_date = "2024-08-01"
    end_date = "2024-08-10"
    
    centroid_lon = -58.4
    centroid_lat = -34.6
    
    print("="*60)
    print("DEBUGGING DATA ACQUISITION ISSUES")
    print(f"Test area: Small region around Buenos Aires")
    print(f"Date range: {start_date} to {end_date}")
    print("="*60)
    
    # Test 1: Weather Data
    print("\n1. Testing Weather Data (NASA POWER)...")
    try:
        weather_acq = WeatherDataAcquisition()
        weather_data = weather_acq.get_weather_data(centroid_lon, centroid_lat, start_date, end_date)
        print(f"✓ Weather data retrieved: {len(weather_data)} records")
        print(f"  Columns: {list(weather_data.columns)}")
    except Exception as e:
        print(f"✗ Weather data failed: {e}")
    
    # Test 2: Soil Data
    print("\n2. Testing Soil Data (SoilGrids)...")
    try:
        soil_acq = SoilDataAcquisition()
        soil_data = soil_acq.get_soil_data(centroid_lon, centroid_lat)
        print(f"✓ Soil data retrieved")
        print(f"  Type: {type(soil_data)}")
    except Exception as e:
        print(f"✗ Soil data failed: {e}")
    
    # Test 3: Satellite Data (this is likely the slow one)
    print("\n3. Testing Satellite Data (Google Earth Engine)...")
    print("   This may take a while...")
    try:
        satellite_acq = SatelliteDataAcquisition()
        
        # Set a timeout or use an even smaller area
        print("   Attempting to get Sentinel-2 data...")
        satellite_data = satellite_acq.get_sentinel2_data(small_geometry, start_date, end_date)
        
        print(f"✓ Satellite data retrieved")
        print(f"  Image count: {satellite_data.get('image_count', 'Unknown')}")
        
        if 'time_series' in satellite_data:
            ts_df = satellite_data['time_series']
            print(f"  Time series records: {len(ts_df)}")
            print(f"  Time series columns: {list(ts_df.columns) if not ts_df.empty else 'No data'}")
        
    except Exception as e:
        print(f"✗ Satellite data failed: {e}")
        print("   This is likely where the system gets stuck.")
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_individual_apis()