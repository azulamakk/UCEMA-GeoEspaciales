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
        """Comprehensive coverage of ALL Argentina agricultural regions"""
        return {
            # BUENOS AIRES PROVINCE - COMPLETE COVERAGE (0.4° x 0.4° grid)
            'buenos_aires_01_northwest': {
                'geometry': [[-61.2, -35.0], [-60.8, -35.0], [-60.8, -34.6], [-61.2, -34.6], [-61.2, -35.0]],
                'name': 'Buenos Aires Zone 1 (Northwest)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'buenos_aires_02_northeast': {
                'geometry': [[-60.8, -35.0], [-60.4, -35.0], [-60.4, -34.6], [-60.8, -34.6], [-60.8, -35.0]],
                'name': 'Buenos Aires Zone 2 (Northeast)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'buenos_aires_03_east': {
                'geometry': [[-60.4, -35.0], [-60.0, -35.0], [-60.0, -34.6], [-60.4, -34.6], [-60.4, -35.0]],
                'name': 'Buenos Aires Zone 3 (East)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'buenos_aires_04_far_east': {
                'geometry': [[-60.0, -35.0], [-59.6, -35.0], [-59.6, -34.6], [-60.0, -34.6], [-60.0, -35.0]],
                'name': 'Buenos Aires Zone 4 (Far East)',
                'crops': ['soybean', 'wheat', 'sunflower'], 'priority': 'high'
            },
            'buenos_aires_05_central_west': {
                'geometry': [[-61.2, -35.4], [-60.8, -35.4], [-60.8, -35.0], [-61.2, -35.0], [-61.2, -35.4]],
                'name': 'Buenos Aires Zone 5 (Central West)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'buenos_aires_06_central': {
                'geometry': [[-60.8, -35.4], [-60.4, -35.4], [-60.4, -35.0], [-60.8, -35.0], [-60.8, -35.4]],
                'name': 'Buenos Aires Zone 6 (Central)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'buenos_aires_07_central_east': {
                'geometry': [[-60.4, -35.4], [-60.0, -35.4], [-60.0, -35.0], [-60.4, -35.0], [-60.4, -35.4]],
                'name': 'Buenos Aires Zone 7 (Central East)',
                'crops': ['soybean', 'corn', 'sunflower'], 'priority': 'high'
            },
            'buenos_aires_08_southwest': {
                'geometry': [[-61.6, -36.0], [-61.2, -36.0], [-61.2, -35.6], [-61.6, -35.6], [-61.6, -36.0]],
                'name': 'Buenos Aires Zone 8 (Southwest)',
                'crops': ['wheat', 'barley', 'sunflower'], 'priority': 'high'
            },
            'buenos_aires_09_south_central': {
                'geometry': [[-61.2, -36.0], [-60.8, -36.0], [-60.8, -35.6], [-61.2, -35.6], [-61.2, -36.0]],
                'name': 'Buenos Aires Zone 9 (South Central)',
                'crops': ['wheat', 'soybean', 'barley'], 'priority': 'high'
            },
            'buenos_aires_10_southeast': {
                'geometry': [[-60.8, -36.0], [-60.4, -36.0], [-60.4, -35.6], [-60.8, -35.6], [-60.8, -36.0]],
                'name': 'Buenos Aires Zone 10 (Southeast)',
                'crops': ['wheat', 'soybean', 'sunflower'], 'priority': 'high'
            },
            
            # CÓRDOBA PROVINCE - COMPLETE COVERAGE
            'cordoba_01_north': {
                'geometry': [[-64.0, -31.5], [-63.6, -31.5], [-63.6, -31.1], [-64.0, -31.1], [-64.0, -31.5]],
                'name': 'Córdoba Zone 1 (North)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'cordoba_02_northeast': {
                'geometry': [[-63.6, -31.5], [-63.2, -31.5], [-63.2, -31.1], [-63.6, -31.1], [-63.6, -31.5]],
                'name': 'Córdoba Zone 2 (Northeast)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'cordoba_03_east': {
                'geometry': [[-63.2, -31.5], [-62.8, -31.5], [-62.8, -31.1], [-63.2, -31.1], [-63.2, -31.5]],
                'name': 'Córdoba Zone 3 (East)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'cordoba_04_central_west': {
                'geometry': [[-64.0, -31.9], [-63.6, -31.9], [-63.6, -31.5], [-64.0, -31.5], [-64.0, -31.9]],
                'name': 'Córdoba Zone 4 (Central West)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'cordoba_05_central': {
                'geometry': [[-63.6, -31.9], [-63.2, -31.9], [-63.2, -31.5], [-63.6, -31.5], [-63.6, -31.9]],
                'name': 'Córdoba Zone 5 (Central)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'cordoba_06_central_east': {
                'geometry': [[-63.2, -31.9], [-62.8, -31.9], [-62.8, -31.5], [-63.2, -31.5], [-63.2, -31.9]],
                'name': 'Córdoba Zone 6 (Central East)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'cordoba_07_south': {
                'geometry': [[-64.0, -32.3], [-63.6, -32.3], [-63.6, -31.9], [-64.0, -31.9], [-64.0, -32.3]],
                'name': 'Córdoba Zone 7 (South)',
                'crops': ['soybean', 'wheat', 'corn'], 'priority': 'high'
            },
            'cordoba_08_southeast': {
                'geometry': [[-63.6, -32.3], [-63.2, -32.3], [-63.2, -31.9], [-63.6, -31.9], [-63.6, -32.3]],
                'name': 'Córdoba Zone 8 (Southeast)',
                'crops': ['soybean', 'wheat', 'sunflower'], 'priority': 'high'
            },
            
            # SANTA FE PROVINCE - COMPLETE COVERAGE
            'santa_fe_01_northwest': {
                'geometry': [[-61.4, -30.4], [-61.0, -30.4], [-61.0, -30.0], [-61.4, -30.0], [-61.4, -30.4]],
                'name': 'Santa Fe Zone 1 (Northwest)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'santa_fe_02_north': {
                'geometry': [[-61.0, -30.4], [-60.6, -30.4], [-60.6, -30.0], [-61.0, -30.0], [-61.0, -30.4]],
                'name': 'Santa Fe Zone 2 (North)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'santa_fe_03_northeast': {
                'geometry': [[-60.6, -30.4], [-60.2, -30.4], [-60.2, -30.0], [-60.6, -30.0], [-60.6, -30.4]],
                'name': 'Santa Fe Zone 3 (Northeast)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'santa_fe_04_central_west': {
                'geometry': [[-61.4, -30.8], [-61.0, -30.8], [-61.0, -30.4], [-61.4, -30.4], [-61.4, -30.8]],
                'name': 'Santa Fe Zone 4 (Central West)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'santa_fe_05_central': {
                'geometry': [[-61.0, -30.8], [-60.6, -30.8], [-60.6, -30.4], [-61.0, -30.4], [-61.0, -30.8]],
                'name': 'Santa Fe Zone 5 (Central)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'santa_fe_06_central_east': {
                'geometry': [[-60.6, -30.8], [-60.2, -30.8], [-60.2, -30.4], [-60.6, -30.4], [-60.6, -30.8]],
                'name': 'Santa Fe Zone 6 (Central East)',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'high'
            },
            'santa_fe_07_south': {
                'geometry': [[-61.4, -31.2], [-61.0, -31.2], [-61.0, -30.8], [-61.4, -30.8], [-61.4, -31.2]],
                'name': 'Santa Fe Zone 7 (South)',
                'crops': ['soybean', 'wheat', 'sunflower'], 'priority': 'high'
            },
            'santa_fe_08_southeast': {
                'geometry': [[-61.0, -31.2], [-60.6, -31.2], [-60.6, -30.8], [-61.0, -30.8], [-61.0, -31.2]],
                'name': 'Santa Fe Zone 8 (Southeast)',
                'crops': ['soybean', 'wheat', 'sunflower'], 'priority': 'high'
            },
            
            # ENTRE RÍOS PROVINCE
            'entre_rios_01_north': {
                'geometry': [[-59.8, -30.8], [-59.4, -30.8], [-59.4, -30.4], [-59.8, -30.4], [-59.8, -30.8]],
                'name': 'Entre Ríos Zone 1 (North)',
                'crops': ['soybean', 'corn', 'rice'], 'priority': 'medium'
            },
            'entre_rios_02_central': {
                'geometry': [[-59.8, -31.2], [-59.4, -31.2], [-59.4, -30.8], [-59.8, -30.8], [-59.8, -31.2]],
                'name': 'Entre Ríos Zone 2 (Central)',
                'crops': ['soybean', 'corn', 'rice'], 'priority': 'medium'
            },
            'entre_rios_03_south': {
                'geometry': [[-59.8, -31.6], [-59.4, -31.6], [-59.4, -31.2], [-59.8, -31.2], [-59.8, -31.6]],
                'name': 'Entre Ríos Zone 3 (South)',
                'crops': ['soybean', 'wheat', 'rice'], 'priority': 'medium'
            },
            
            # LA PAMPA PROVINCE
            'la_pampa_01_east': {
                'geometry': [[-65.2, -36.4], [-64.8, -36.4], [-64.8, -36.0], [-65.2, -36.0], [-65.2, -36.4]],
                'name': 'La Pampa Zone 1 (East)',
                'crops': ['wheat', 'sunflower', 'corn'], 'priority': 'medium'
            },
            'la_pampa_02_central': {
                'geometry': [[-65.6, -36.4], [-65.2, -36.4], [-65.2, -36.0], [-65.6, -36.0], [-65.6, -36.4]],
                'name': 'La Pampa Zone 2 (Central)',
                'crops': ['wheat', 'sunflower', 'barley'], 'priority': 'medium'
            },
            'la_pampa_03_north': {
                'geometry': [[-65.2, -36.0], [-64.8, -36.0], [-64.8, -35.6], [-65.2, -35.6], [-65.2, -36.0]],
                'name': 'La Pampa Zone 3 (North)',
                'crops': ['wheat', 'sunflower', 'corn'], 'priority': 'medium'
            },
            
            # NORTHERN ARGENTINA - EXTENDED COVERAGE
            'santiago_estero_01': {
                'geometry': [[-63.4, -28.0], [-63.0, -28.0], [-63.0, -27.6], [-63.4, -27.6], [-63.4, -28.0]],
                'name': 'Santiago del Estero Zone 1',
                'crops': ['soybean', 'cotton', 'wheat'], 'priority': 'medium'
            },
            'santiago_estero_02': {
                'geometry': [[-63.0, -28.0], [-62.6, -28.0], [-62.6, -27.6], [-63.0, -27.6], [-63.0, -28.0]],
                'name': 'Santiago del Estero Zone 2',
                'crops': ['soybean', 'cotton', 'wheat'], 'priority': 'medium'
            },
            'chaco_01_south': {
                'geometry': [[-60.8, -27.0], [-60.4, -27.0], [-60.4, -26.6], [-60.8, -26.6], [-60.8, -27.0]],
                'name': 'Chaco Zone 1 (South)',
                'crops': ['cotton', 'soybean', 'sunflower'], 'priority': 'medium'
            },
            'chaco_02_central': {
                'geometry': [[-60.4, -27.0], [-60.0, -27.0], [-60.0, -26.6], [-60.4, -26.6], [-60.4, -27.0]],
                'name': 'Chaco Zone 2 (Central)',
                'crops': ['cotton', 'soybean', 'sunflower'], 'priority': 'medium'
            },
            'tucuman_01': {
                'geometry': [[-65.6, -26.8], [-65.2, -26.8], [-65.2, -26.4], [-65.6, -26.4], [-65.6, -26.8]],
                'name': 'Tucumán Zone 1',
                'crops': ['sugarcane', 'soybean', 'citrus'], 'priority': 'low'
            },
            'salta_01': {
                'geometry': [[-65.2, -25.2], [-64.8, -25.2], [-64.8, -24.8], [-65.2, -24.8], [-65.2, -25.2]],
                'name': 'Salta Zone 1',
                'crops': ['soybean', 'sugarcane', 'beans'], 'priority': 'low'
            },
            
            # TEST AREAS
            'test_area_micro': {
                'geometry': [[-60.0, -34.0], [-59.8, -34.0], [-59.8, -33.8], [-60.0, -33.8], [-60.0, -34.0]],
                'name': 'Micro Test Area',
                'crops': ['soybean', 'corn', 'wheat'], 'priority': 'test'
            }
        }

# Global configuration instance
api_config = APIConfig()