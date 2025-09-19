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
        """Default study areas for Argentine agricultural regions"""
        return {
            'test_area_small': {
                'geometry': [[-58.7, -34.8], [-58.3, -34.8], [-58.3, -34.4], [-58.7, -34.4], [-58.7, -34.8]],
                'name': 'Small Test Area (Buenos Aires Region)',
                'crops': ['soybean', 'corn', 'wheat']
            },
            'pampas_region': {
                'geometry': [[-68.0, -40.0], [-57.0, -40.0], [-57.0, -30.0], [-68.0, -30.0], [-68.0, -40.0]],
                'name': 'Pampas Agricultural Region (Large - may hit EE limits)',
                'crops': ['soybean', 'corn', 'wheat']
            },
            'cordoba_small': {
                'geometry': [[-64.5, -32.0], [-64.0, -32.0], [-64.0, -31.5], [-64.5, -31.5], [-64.5, -32.0]],
                'name': 'Small Córdoba Test Area',
                'crops': ['soybean', 'corn', 'wheat']
            },
            'buenos_aires_small': {
                'geometry': [[-59.0, -36.0], [-58.5, -36.0], [-58.5, -35.5], [-59.0, -35.5], [-59.0, -36.0]],
                'name': 'Small Buenos Aires Test Area',
                'crops': ['soybean', 'corn', 'wheat']
            }
        }

# Global configuration instance
api_config = APIConfig()