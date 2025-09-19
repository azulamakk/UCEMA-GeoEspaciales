"""
Test script to verify API connectivity for water stress detection system
"""
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.api_config import api_config
import ee
import requests

def test_google_earth_engine():
    """Test Google Earth Engine connectivity"""
    try:
        ee.Initialize()
        print("✓ Google Earth Engine: Successfully initialized")
        
        # Test basic functionality
        image = ee.Image('COPERNICUS/S2_SR/20210109T170219_20210109T170218_T14SQH')
        info = image.getInfo()
        print("✓ Google Earth Engine: Successfully accessed Sentinel-2 data")
        return True
    except Exception as e:
        print(f"✗ Google Earth Engine: {e}")
        return False

def test_nasa_power():
    """Test NASA POWER API connectivity"""
    try:
        # Test basic API request
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            'parameters': 'T2M',
            'community': 'AG',
            'longitude': -60.0,
            'latitude': -35.0,
            'start': '20240101',
            'end': '20240102',
            'format': 'JSON'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            print("✓ NASA POWER: Successfully connected and retrieved data")
            return True
        else:
            print(f"✗ NASA POWER: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ NASA POWER: {e}")
        return False

def test_soilgrids():
    """Test SoilGrids API connectivity"""
    try:
        # Test basic API request
        url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        params = {
            'lon': -60.0,
            'lat': -35.0,
            'property': 'bdod',
            'depth': '0-5cm',
            'value': 'mean'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            print("✓ SoilGrids: Successfully connected and retrieved data")
            return True
        else:
            print(f"✗ SoilGrids: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ SoilGrids: {e}")
        return False

def main():
    """Run all API connectivity tests"""
    print("Testing API connectivity for Water Stress Detection System\n")
    
    results = []
    
    print("1. Testing Google Earth Engine...")
    results.append(test_google_earth_engine())
    
    print("\n2. Testing NASA POWER API...")
    results.append(test_nasa_power())
    
    print("\n3. Testing SoilGrids API...")
    results.append(test_soilgrids())
    
    print("\n" + "="*50)
    print("SUMMARY:")
    if all(results):
        print("✓ All APIs are accessible! System is ready to run.")
        return 0
    else:
        print("✗ Some APIs are not accessible. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())