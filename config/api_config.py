"""
Configuration management for API keys and endpoints
"""
import os
from dotenv import load_dotenv
from typing import Dict, Optional

load_dotenv()

class APIConfig:
    """Manages API configurations and authentication"""
    
    def __init__(self):
        self.google_earth_engine = {
            'service_account_key': os.getenv('GEE_SERVICE_ACCOUNT_KEY'),
            'project_id': os.getenv('GEE_PROJECT_ID'),
            'use_service_account': os.getenv('GEE_USE_SERVICE_ACCOUNT', 'true').lower() == 'true'
        }
        
        self.nasa_power = {
            'base_url': 'https://power.larc.nasa.gov/api/temporal/daily/point',
            'parameters': ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'EVPTRNS'],
            'community': 'AG'  # Agroclimatology community
        }
        
        self.soilgrids = {
            'base_url': 'https://rest.isric.org/soilgrids/v2.0/properties/query',
            'properties': ['bdod', 'cec', 'clay', 'nitrogen', 'ocd', 'ocs', 'phh2o', 'sand', 'silt', 'soc'],
            'depths': ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
        }
    
    def validate_credentials(self) -> Dict[str, bool]:
        """Validate that required credentials are available"""
        validation = {
            'google_earth_engine': bool(self.google_earth_engine.get('service_account_key')),
            'nasa_power': True,  # No API key required
            'soilgrids': True    # No API key required
        }
        return validation
    
    def get_study_area_argentina(self) -> Dict:
        """Comprehensive study areas for Argentine agricultural regions"""
        return {
            # PRIMARY AGRICULTURAL REGIONS (Optimized sizes for Google Earth Engine)
            'pampas_central': {
                'geometry': [[-63.0, -36.0], [-59.0, -36.0], [-59.0, -32.0], [-63.0, -32.0], [-63.0, -36.0]],
                'name': 'Central Pampas (Buenos Aires/Córdoba Core)',
                'crops': ['soybean', 'corn', 'wheat'],
                'priority': 'high'
            },
            'cordoba_agriculture': {
                'geometry': [[-65.0, -33.0], [-62.0, -33.0], [-62.0, -30.0], [-65.0, -30.0], [-65.0, -33.0]],
                'name': 'Córdoba Agricultural Zone',
                'crops': ['soybean', 'corn', 'wheat'],
                'priority': 'high'
            },
            'santa_fe_agriculture': {
                'geometry': [[-62.0, -33.0], [-59.0, -33.0], [-59.0, -29.0], [-62.0, -29.0], [-62.0, -33.0]],
                'name': 'Santa Fe Agricultural Zone',
                'crops': ['soybean', 'corn', 'wheat'],
                'priority': 'high'
            },
            'buenos_aires_north': {
                'geometry': [[-62.0, -36.0], [-58.0, -36.0], [-58.0, -33.0], [-62.0, -33.0], [-62.0, -36.0]],
                'name': 'Northern Buenos Aires Agricultural Zone',
                'crops': ['soybean', 'corn', 'wheat'],
                'priority': 'high'
            },
            'buenos_aires_south': {
                'geometry': [[-63.0, -39.0], [-58.0, -39.0], [-58.0, -36.0], [-63.0, -36.0], [-63.0, -39.0]],
                'name': 'Southern Buenos Aires Agricultural Zone',
                'crops': ['wheat', 'barley', 'canola'],
                'priority': 'medium'
            },
            'entre_rios': {
                'geometry': [[-60.0, -33.0], [-57.5, -33.0], [-57.5, -30.0], [-60.0, -30.0], [-60.0, -33.0]],
                'name': 'Entre Ríos Agricultural Zone',
                'crops': ['soybean', 'corn', 'rice'],
                'priority': 'medium'
            },
            'la_pampa': {
                'geometry': [[-68.0, -38.0], [-64.0, -38.0], [-64.0, -35.0], [-68.0, -35.0], [-68.0, -38.0]],
                'name': 'La Pampa Agricultural Zone',
                'crops': ['wheat', 'sunflower', 'corn'],
                'priority': 'medium'
            },
            
            # NORTHERN AGRICULTURAL REGIONS
            'santiago_estero': {
                'geometry': [[-65.0, -29.0], [-62.0, -29.0], [-62.0, -26.0], [-65.0, -26.0], [-65.0, -29.0]],
                'name': 'Santiago del Estero Agricultural Zone',
                'crops': ['soybean', 'cotton', 'wheat'],
                'priority': 'medium'
            },
            'tucuman_salta': {
                'geometry': [[-66.0, -27.0], [-64.0, -27.0], [-64.0, -24.0], [-66.0, -24.0], [-66.0, -27.0]],
                'name': 'Tucumán-Salta Agricultural Zone',
                'crops': ['sugarcane', 'soybean', 'citrus'],
                'priority': 'low'
            },
            'chaco': {
                'geometry': [[-62.0, -28.0], [-59.0, -28.0], [-59.0, -25.0], [-62.0, -25.0], [-62.0, -28.0]],
                'name': 'Chaco Agricultural Zone',
                'crops': ['cotton', 'soybean', 'sunflower'],
                'priority': 'low'
            },
            
            # COMPREHENSIVE NATIONAL ANALYSIS
            'argentina_complete': {
                'geometry': [[-68.0, -40.0], [-57.0, -40.0], [-57.0, -24.0], [-68.0, -24.0], [-68.0, -40.0]],
                'name': 'Complete Argentina Agricultural Analysis',
                'crops': ['soybean', 'corn', 'wheat', 'sunflower', 'barley'],
                'priority': 'national',
                'warning': 'Large area - may require special processing'
            },
            
            # LEGACY/TEST AREAS (kept for compatibility)
            'test_area_small': {
                'geometry': [[-60.0, -34.0], [-59.5, -34.0], [-59.5, -33.5], [-60.0, -33.5], [-60.0, -34.0]],
                'name': 'Small Test Area (Agricultural Focus)',
                'crops': ['soybean', 'corn', 'wheat'],
                'priority': 'test'
            }
        }

# Global configuration instance
api_config = APIConfig()