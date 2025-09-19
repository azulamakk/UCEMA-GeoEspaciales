"""
Prescription map generation for variable rate irrigation
Creates spatially explicit irrigation recommendations based on water stress analysis
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
from datetime import datetime

class PrescriptionMapGenerator:
    """Generate variable rate irrigation prescription maps"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.irrigation_rates = self._initialize_irrigation_rates()
        
    def _initialize_irrigation_rates(self) -> Dict[str, Dict[str, float]]:
        """Initialize irrigation rate recommendations by crop and stress level"""
        return {
            'soybean': {
                'no_stress': 0.0,      # mm/day
                'mild_stress': 3.0,
                'moderate_stress': 5.0,
                'severe_stress': 8.0,
                'critical_stress': 12.0
            },
            'corn': {
                'no_stress': 0.0,
                'mild_stress': 4.0,
                'moderate_stress': 7.0,
                'severe_stress': 10.0,
                'critical_stress': 15.0
            },
            'wheat': {
                'no_stress': 0.0,
                'mild_stress': 2.0,
                'moderate_stress': 4.0,
                'severe_stress': 6.0,
                'critical_stress': 10.0
            }
        }
    
    def create_prescription_map(self, 
                               stress_data: pd.DataFrame,
                               geometry: Union[List[List[float]], Polygon],
                               crop_type: str = 'soybean',
                               resolution: float = 30.0,
                               irrigation_efficiency: float = 0.8) -> Dict[str, Any]:
        """
        Create a prescription map for variable rate irrigation
        
        Args:
            stress_data: DataFrame with spatial stress data
            geometry: Field boundary geometry
            crop_type: Type of crop
            resolution: Spatial resolution in meters
            irrigation_efficiency: Irrigation system efficiency (0-1)
            
        Returns:
            Dictionary containing prescription map data and metadata
        """
        try:
            # Validate inputs
            if crop_type not in self.irrigation_rates:
                self.logger.warning(f"Unknown crop type: {crop_type}. Using soybean rates.")
                crop_type = 'soybean'
            
            # Convert geometry to polygon if needed
            if isinstance(geometry, list):
                field_polygon = Polygon(geometry[0] if isinstance(geometry[0][0], list) else geometry)
            else:
                field_polygon = geometry
            
            # Calculate field bounds
            bounds = field_polygon.bounds
            
            # Create spatial grid
            grid_data = self._create_spatial_grid(bounds, resolution)
            
            # Interpolate stress values to grid
            stress_grid = self._interpolate_stress_to_grid(stress_data, grid_data)
            
            # Calculate irrigation recommendations
            irrigation_grid = self._calculate_irrigation_rates(
                stress_grid, crop_type, irrigation_efficiency
            )
            
            # Create prescription zones
            zones = self._create_prescription_zones(irrigation_grid)
            
            # Generate map outputs
            map_outputs = self._generate_map_outputs(
                grid_data, stress_grid, irrigation_grid, zones, field_polygon
            )
            
            # Calculate summary statistics
            summary_stats = self._calculate_prescription_summary(irrigation_grid, zones)
            
            prescription_map = {
                'timestamp': datetime.now().isoformat(),
                'crop_type': crop_type,
                'field_area_ha': field_polygon.area / 10000,  # Convert to hectares
                'resolution_m': resolution,
                'irrigation_efficiency': irrigation_efficiency,
                'stress_grid': stress_grid,
                'irrigation_grid': irrigation_grid,
                'prescription_zones': zones,
                'summary_statistics': summary_stats,
                'map_outputs': map_outputs,
                'metadata': {
                    'bounds': bounds,
                    'grid_shape': grid_data['shape'],
                    'total_points': len(stress_data)
                }
            }
            
            self.logger.info(f"Prescription map created for {field_polygon.area/10000:.2f} ha field")
            
            return prescription_map
            
        except Exception as e:
            self.logger.error(f"Error creating prescription map: {e}")
            raise
    
    def _create_spatial_grid(self, bounds: Tuple[float, float, float, float], 
                           resolution: float) -> Dict[str, Any]:
        """Create spatial grid for the field"""
        try:
            minx, miny, maxx, maxy = bounds
            
            # Calculate grid dimensions
            width = int((maxx - minx) / resolution) + 1
            height = int((maxy - miny) / resolution) + 1
            
            # Create coordinate arrays
            x_coords = np.linspace(minx, maxx, width)
            y_coords = np.linspace(miny, maxy, height)
            
            # Create meshgrid
            X, Y = np.meshgrid(x_coords, y_coords)
            
            # Create transform for rasterio
            transform = from_bounds(minx, miny, maxx, maxy, width, height)
            
            return {
                'x_coords': x_coords,
                'y_coords': y_coords,
                'X': X,
                'Y': Y,
                'shape': (height, width),
                'transform': transform,
                'bounds': bounds
            }
            
        except Exception as e:
            self.logger.error(f"Error creating spatial grid: {e}")
            raise
    
    def _interpolate_stress_to_grid(self, stress_data: pd.DataFrame, 
                                   grid_data: Dict[str, Any]) -> np.ndarray:
        """Interpolate point stress data to grid"""
        try:
            from scipy.interpolate import griddata
            
            # Extract coordinates and stress values
            if 'longitude' in stress_data.columns and 'latitude' in stress_data.columns:
                points = stress_data[['longitude', 'latitude']].values
            elif 'lon' in stress_data.columns and 'lat' in stress_data.columns:
                points = stress_data[['lon', 'lat']].values
            else:
                # Assume first two columns are coordinates
                points = stress_data.iloc[:, :2].values
            
            # Determine stress values
            stress_column = self._identify_stress_column(stress_data)
            stress_values = stress_data[stress_column].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(stress_values)
            points = points[valid_mask]
            stress_values = stress_values[valid_mask]
            
            if len(points) < 3:
                self.logger.warning("Insufficient data points for interpolation")
                return np.full(grid_data['shape'], 0.5)  # Default moderate stress
            
            # Interpolate to grid
            grid_points = np.column_stack([grid_data['X'].ravel(), grid_data['Y'].ravel()])
            
            # Use different interpolation methods
            try:
                # Try cubic interpolation first
                stress_grid = griddata(
                    points, stress_values, grid_points, method='cubic', fill_value=np.nan
                )
            except:
                # Fall back to linear interpolation
                stress_grid = griddata(
                    points, stress_values, grid_points, method='linear', fill_value=np.nan
                )
            
            # Fill remaining NaN values with nearest neighbor
            nan_mask = np.isnan(stress_grid)
            if nan_mask.any():
                stress_grid[nan_mask] = griddata(
                    points, stress_values, grid_points[nan_mask], 
                    method='nearest'
                )
            
            # Reshape to grid
            stress_grid = stress_grid.reshape(grid_data['shape'])
            
            # Ensure values are in valid range [0, 1]
            stress_grid = np.clip(stress_grid, 0, 1)
            
            return stress_grid
            
        except Exception as e:
            self.logger.error(f"Error interpolating stress data: {e}")
            # Return default grid with moderate stress
            return np.full(grid_data['shape'], 0.5)
    
    def _identify_stress_column(self, df: pd.DataFrame) -> str:
        """Identify the stress indicator column"""
        stress_columns = [
            'water_stress_level', 'stress_level', 'combined_stress_index',
            'alert_score', 'CWSI', 'stress_index'
        ]
        
        for col in stress_columns:
            if col in df.columns:
                return col
        
        # Look for columns with 'stress' in the name
        stress_cols = [col for col in df.columns if 'stress' in col.lower()]
        if stress_cols:
            return stress_cols[0]
        
        # Default to last numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[-1]
        
        raise ValueError("No suitable stress indicator column found")
    
    def _calculate_irrigation_rates(self, stress_grid: np.ndarray, 
                                   crop_type: str, 
                                   efficiency: float) -> np.ndarray:
        """Calculate irrigation rates based on stress levels"""
        try:
            rates = self.irrigation_rates[crop_type]
            irrigation_grid = np.zeros_like(stress_grid)
            
            # Define stress level thresholds
            thresholds = {
                'no_stress': (0.0, 0.1),
                'mild_stress': (0.1, 0.3),
                'moderate_stress': (0.3, 0.6),
                'severe_stress': (0.6, 0.8),
                'critical_stress': (0.8, 1.0)
            }
            
            # Apply rates based on stress levels
            for stress_level, (min_val, max_val) in thresholds.items():
                mask = (stress_grid >= min_val) & (stress_grid < max_val)
                irrigation_rate = rates[stress_level] / efficiency  # Adjust for efficiency
                irrigation_grid[mask] = irrigation_rate
            
            # Handle maximum stress level
            mask = stress_grid >= 0.8
            irrigation_grid[mask] = rates['critical_stress'] / efficiency
            
            return irrigation_grid
            
        except Exception as e:
            self.logger.error(f"Error calculating irrigation rates: {e}")
            return np.zeros_like(stress_grid)
    
    def _create_prescription_zones(self, irrigation_grid: np.ndarray) -> Dict[str, Any]:
        """Create prescription zones based on irrigation rates"""
        try:
            # Define irrigation zones
            zone_definitions = {
                1: {'name': 'No Irrigation', 'min_rate': 0.0, 'max_rate': 1.0, 'color': '#2E8B57'},
                2: {'name': 'Light Irrigation', 'min_rate': 1.0, 'max_rate': 4.0, 'color': '#90EE90'},
                3: {'name': 'Moderate Irrigation', 'min_rate': 4.0, 'max_rate': 8.0, 'color': '#FFD700'},
                4: {'name': 'Heavy Irrigation', 'min_rate': 8.0, 'max_rate': 12.0, 'color': '#FF8C00'},
                5: {'name': 'Critical Irrigation', 'min_rate': 12.0, 'max_rate': 20.0, 'color': '#FF4500'}
            }
            
            # Create zone grid
            zone_grid = np.zeros_like(irrigation_grid, dtype=int)
            
            for zone_id, zone_def in zone_definitions.items():
                mask = ((irrigation_grid >= zone_def['min_rate']) & 
                       (irrigation_grid < zone_def['max_rate']))
                zone_grid[mask] = zone_id
            
            # Calculate zone statistics
            zone_stats = {}
            total_pixels = irrigation_grid.size
            
            for zone_id, zone_def in zone_definitions.items():
                zone_pixels = (zone_grid == zone_id).sum()
                zone_area_pct = (zone_pixels / total_pixels) * 100
                avg_irrigation_rate = irrigation_grid[zone_grid == zone_id].mean() if zone_pixels > 0 else 0
                
                zone_stats[zone_id] = {
                    'name': zone_def['name'],
                    'area_percentage': zone_area_pct,
                    'pixel_count': zone_pixels,
                    'avg_irrigation_rate': avg_irrigation_rate,
                    'color': zone_def['color']
                }
            
            return {
                'zone_grid': zone_grid,
                'zone_definitions': zone_definitions,
                'zone_statistics': zone_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error creating prescription zones: {e}")
            return {'zone_grid': np.zeros_like(irrigation_grid), 'zone_definitions': {}, 'zone_statistics': {}}
    
    def _generate_map_outputs(self, grid_data: Dict[str, Any],
                             stress_grid: np.ndarray,
                             irrigation_grid: np.ndarray,
                             zones: Dict[str, Any],
                             field_polygon: Polygon) -> Dict[str, Any]:
        """Generate various map outputs"""
        try:
            outputs = {}
            
            # Create interactive folium map
            outputs['folium_map'] = self._create_folium_map(
                grid_data, stress_grid, irrigation_grid, zones, field_polygon
            )
            
            # Create static matplotlib plots
            outputs['static_plots'] = self._create_static_plots(
                grid_data, stress_grid, irrigation_grid, zones
            )
            
            # Create GeoTIFF outputs (metadata only, actual files would be saved separately)
            outputs['geotiff_metadata'] = {
                'stress_map': {
                    'filename': 'water_stress_map.tif',
                    'description': 'Water stress intensity map',
                    'units': 'stress_index_0_1',
                    'transform': grid_data['transform']
                },
                'irrigation_map': {
                    'filename': 'irrigation_prescription_map.tif',
                    'description': 'Irrigation rate prescription map',
                    'units': 'mm_per_day',
                    'transform': grid_data['transform']
                },
                'zone_map': {
                    'filename': 'irrigation_zones_map.tif',
                    'description': 'Irrigation management zones',
                    'units': 'zone_id',
                    'transform': grid_data['transform']
                }
            }
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"Error generating map outputs: {e}")
            return {}
    
    def _create_folium_map(self, grid_data: Dict[str, Any],
                          stress_grid: np.ndarray,
                          irrigation_grid: np.ndarray,
                          zones: Dict[str, Any],
                          field_polygon: Polygon) -> str:
        """Create interactive folium map"""
        try:
            # Calculate map center
            bounds = field_polygon.bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=15,
                tiles='OpenStreetMap'
            )
            
            # Add field boundary
            field_coords = [[lat, lon] for lon, lat in field_polygon.exterior.coords]
            folium.Polygon(
                locations=field_coords,
                color='black',
                weight=3,
                fillOpacity=0,
                popup='Field Boundary'
            ).add_to(m)
            
            # Create heat map for stress levels
            X, Y = grid_data['X'], grid_data['Y']
            heat_data = []
            
            for i in range(0, X.shape[0], 2):  # Subsample for performance
                for j in range(0, X.shape[1], 2):
                    lat, lon = Y[i, j], X[i, j]
                    intensity = stress_grid[i, j]
                    heat_data.append([lat, lon, intensity])
            
            HeatMap(
                heat_data,
                name='Water Stress Intensity',
                radius=20,
                max_zoom=18,
                show=True
            ).add_to(m)
            
            # Add irrigation zones as colored polygons
            self._add_irrigation_zones_to_map(m, grid_data, zones)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add legend
            self._add_map_legend(m, zones)
            
            # Return HTML representation
            return m._repr_html_()
            
        except Exception as e:
            self.logger.error(f"Error creating folium map: {e}")
            return "<p>Error generating interactive map</p>"
    
    def _add_irrigation_zones_to_map(self, map_obj: folium.Map,
                                   grid_data: Dict[str, Any],
                                   zones: Dict[str, Any]):
        """Add irrigation zones as colored polygons to folium map"""
        try:
            zone_grid = zones['zone_grid']
            zone_defs = zones['zone_definitions']
            X, Y = grid_data['X'], grid_data['Y']
            
            # Sample grid points for polygon creation (reduce density for performance)
            step = max(1, min(zone_grid.shape) // 20)
            
            for zone_id, zone_def in zone_defs.items():
                if zone_id in zones['zone_statistics'] and zones['zone_statistics'][zone_id]['pixel_count'] > 0:
                    # Find zone pixels
                    zone_mask = zone_grid == zone_id
                    
                    # Create simplified polygons for each contiguous zone area
                    zone_coords = []
                    for i in range(0, zone_mask.shape[0], step):
                        for j in range(0, zone_mask.shape[1], step):
                            if zone_mask[i, j]:
                                lat, lon = Y[i, j], X[i, j]
                                zone_coords.append([lat, lon])
                    
                    if zone_coords:
                        # Create point markers for zones (simplified representation)
                        for coord in zone_coords[::5]:  # Further subsample
                            folium.CircleMarker(
                                location=coord,
                                radius=3,
                                color=zone_def['color'],
                                fillColor=zone_def['color'],
                                fillOpacity=0.6,
                                popup=f"{zone_def['name']}<br>Rate: {zones['zone_statistics'][zone_id]['avg_irrigation_rate']:.1f} mm/day"
                            ).add_to(map_obj)
            
        except Exception as e:
            self.logger.error(f"Error adding irrigation zones to map: {e}")
    
    def _add_map_legend(self, map_obj: folium.Map, zones: Dict[str, Any]):
        """Add legend to folium map"""
        try:
            legend_html = '''
            <div style="position: fixed; 
                        top: 10px; left: 50px; width: 200px; height: 120px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <h4>Irrigation Zones</h4>
            '''
            
            for zone_id, zone_stats in zones['zone_statistics'].items():
                if zone_stats['pixel_count'] > 0:
                    zone_def = zones['zone_definitions'][zone_id]
                    legend_html += f'''
                    <p><span style="color:{zone_def['color']};">●</span> {zone_def['name']}<br>
                    &nbsp;&nbsp;{zone_stats['area_percentage']:.1f}% of field</p>
                    '''
            
            legend_html += '</div>'
            
            map_obj.get_root().html.add_child(folium.Element(legend_html))
            
        except Exception as e:
            self.logger.error(f"Error adding map legend: {e}")
    
    def _create_static_plots(self, grid_data: Dict[str, Any],
                           stress_grid: np.ndarray,
                           irrigation_grid: np.ndarray,
                           zones: Dict[str, Any]) -> Dict[str, str]:
        """Create static matplotlib plots"""
        try:
            plots = {}
            
            # Stress map plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            im = ax.imshow(stress_grid, cmap='RdYlBu_r', extent=grid_data['bounds'])
            ax.set_title('Water Stress Map')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax, label='Stress Index (0-1)')
            plots['stress_map'] = self._plot_to_base64(fig)
            plt.close(fig)
            
            # Irrigation map plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            im = ax.imshow(irrigation_grid, cmap='Blues', extent=grid_data['bounds'])
            ax.set_title('Irrigation Prescription Map')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax, label='Irrigation Rate (mm/day)')
            plots['irrigation_map'] = self._plot_to_base64(fig)
            plt.close(fig)
            
            # Zone statistics plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Zone area distribution
            zone_names = [zones['zone_definitions'][zid]['name'] for zid in zones['zone_statistics'].keys() 
                         if zones['zone_statistics'][zid]['pixel_count'] > 0]
            zone_areas = [zones['zone_statistics'][zid]['area_percentage'] for zid in zones['zone_statistics'].keys() 
                         if zones['zone_statistics'][zid]['pixel_count'] > 0]
            zone_colors = [zones['zone_definitions'][zid]['color'] for zid in zones['zone_statistics'].keys() 
                          if zones['zone_statistics'][zid]['pixel_count'] > 0]
            
            if zone_names:
                ax1.pie(zone_areas, labels=zone_names, colors=zone_colors, autopct='%1.1f%%')
                ax1.set_title('Field Area by Irrigation Zone')
                
                # Irrigation rate by zone
                zone_rates = [zones['zone_statistics'][zid]['avg_irrigation_rate'] for zid in zones['zone_statistics'].keys() 
                             if zones['zone_statistics'][zid]['pixel_count'] > 0]
                bars = ax2.bar(zone_names, zone_rates, color=zone_colors)
                ax2.set_title('Average Irrigation Rate by Zone')
                ax2.set_ylabel('Irrigation Rate (mm/day)')
                ax2.tick_params(axis='x', rotation=45)
            
            plots['zone_statistics'] = self._plot_to_base64(fig)
            plt.close(fig)
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error creating static plots: {e}")
            return {}
    
    def _plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        try:
            import io
            import base64
            
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            self.logger.error(f"Error converting plot to base64: {e}")
            return ""
    
    def _calculate_prescription_summary(self, irrigation_grid: np.ndarray,
                                      zones: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for the prescription map"""
        try:
            # Basic irrigation statistics
            total_irrigation = np.sum(irrigation_grid)
            mean_irrigation = np.mean(irrigation_grid)
            max_irrigation = np.max(irrigation_grid)
            area_requiring_irrigation = (irrigation_grid > 0).sum()
            total_area = irrigation_grid.size
            
            # Zone-based statistics
            zone_summary = {}
            for zone_id, zone_stats in zones['zone_statistics'].items():
                if zone_stats['pixel_count'] > 0:
                    zone_summary[zone_stats['name']] = {
                        'area_percentage': zone_stats['area_percentage'],
                        'avg_irrigation_rate': zone_stats['avg_irrigation_rate']
                    }
            
            # Water application efficiency metrics
            uniform_rate = mean_irrigation  # Uniform application rate
            variable_rate_savings = max(0, (uniform_rate * total_area - total_irrigation) / (uniform_rate * total_area) * 100)
            
            summary = {
                'total_irrigation_volume_per_day': total_irrigation,
                'average_irrigation_rate': mean_irrigation,
                'maximum_irrigation_rate': max_irrigation,
                'percentage_area_requiring_irrigation': (area_requiring_irrigation / total_area) * 100,
                'zone_breakdown': zone_summary,
                'variable_rate_water_savings_percent': variable_rate_savings,
                'irrigation_uniformity_coefficient': self._calculate_uniformity_coefficient(irrigation_grid)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error calculating prescription summary: {e}")
            return {}
    
    def _calculate_uniformity_coefficient(self, irrigation_grid: np.ndarray) -> float:
        """Calculate irrigation uniformity coefficient"""
        try:
            # Christiansen's Uniformity Coefficient
            mean_rate = np.mean(irrigation_grid)
            if mean_rate == 0:
                return 1.0
            
            deviations = np.abs(irrigation_grid - mean_rate)
            sum_deviations = np.sum(deviations)
            n = irrigation_grid.size
            
            uc = 1 - (sum_deviations / (n * mean_rate))
            return max(0, min(1, uc))
            
        except Exception:
            return 0.0
    
    def save_prescription_maps(self, prescription_map: Dict[str, Any], 
                              output_dir: str):
        """Save prescription maps to files"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Save GeoTIFF files
            for map_type, metadata in prescription_map['map_outputs']['geotiff_metadata'].items():
                filename = os.path.join(output_dir, metadata['filename'])
                
                if map_type == 'stress_map':
                    data = prescription_map['stress_grid']
                elif map_type == 'irrigation_map':
                    data = prescription_map['irrigation_grid']
                elif map_type == 'zone_map':
                    data = prescription_map['prescription_zones']['zone_grid']
                else:
                    continue
                
                # Save as GeoTIFF
                with rasterio.open(
                    filename, 'w',
                    driver='GTiff',
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    dtype=data.dtype,
                    crs='EPSG:4326',
                    transform=metadata['transform']
                ) as dst:
                    dst.write(data, 1)
                    dst.set_band_description(1, metadata['description'])
            
            # Save interactive map
            folium_html = prescription_map['map_outputs'].get('folium_map', '')
            if folium_html:
                with open(os.path.join(output_dir, 'interactive_map.html'), 'w') as f:
                    f.write(folium_html)
            
            # Save summary JSON
            summary_data = {
                'timestamp': prescription_map['timestamp'],
                'crop_type': prescription_map['crop_type'],
                'field_area_ha': prescription_map['field_area_ha'],
                'summary_statistics': prescription_map['summary_statistics'],
                'prescription_zones': {
                    'zone_definitions': prescription_map['prescription_zones']['zone_definitions'],
                    'zone_statistics': prescription_map['prescription_zones']['zone_statistics']
                }
            }
            
            with open(os.path.join(output_dir, 'prescription_summary.json'), 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            self.logger.info(f"Prescription maps saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving prescription maps: {e}")
            raise
    
    def create_application_map(self, prescription_map: Dict[str, Any],
                              equipment_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create equipment-specific application map
        
        Args:
            prescription_map: Generated prescription map
            equipment_params: Equipment specifications (width, speed, etc.)
            
        Returns:
            Equipment-specific application instructions
        """
        try:
            # Default equipment parameters
            default_params = {
                'boom_width_m': 24,
                'application_speed_kmh': 15,
                'nozzle_spacing_m': 0.5,
                'min_application_rate': 0.5,
                'max_application_rate': 20.0
            }
            
            # Merge with provided parameters
            params = {**default_params, **equipment_params}
            
            # Convert irrigation rates to equipment settings
            irrigation_grid = prescription_map['irrigation_grid']
            
            # Constrain rates to equipment limits
            constrained_grid = np.clip(
                irrigation_grid, 
                params['min_application_rate'], 
                params['max_application_rate']
            )
            
            # Calculate application time and volume
            field_area_ha = prescription_map['field_area_ha']
            total_volume_needed = np.sum(constrained_grid) * field_area_ha / irrigation_grid.size
            
            # Calculate equipment settings
            application_map = {
                'equipment_parameters': params,
                'application_grid': constrained_grid,
                'total_volume_needed_liters': total_volume_needed * 10000,  # Convert to liters
                'estimated_application_time_hours': field_area_ha / (params['boom_width_m'] * params['application_speed_kmh'] / 1000),
                'variable_rate_zones': self._create_equipment_zones(constrained_grid, params),
                'quality_metrics': {
                    'rate_achievability': np.mean(irrigation_grid <= params['max_application_rate']) * 100,
                    'coverage_efficiency': np.mean(constrained_grid > 0) * 100
                }
            }
            
            return application_map
            
        except Exception as e:
            self.logger.error(f"Error creating application map: {e}")
            return {}
    
    def _create_equipment_zones(self, application_grid: np.ndarray,
                               equipment_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create equipment-specific application zones"""
        try:
            # Simplified zone creation based on application rates
            # In practice, this would consider equipment capabilities more thoroughly
            
            zones = {}
            unique_rates = np.unique(application_grid)
            
            for i, rate in enumerate(unique_rates):
                if rate > 0:
                    zone_mask = application_grid == rate
                    zones[f'zone_{i+1}'] = {
                        'application_rate': float(rate),
                        'area_pixels': int(zone_mask.sum()),
                        'equipment_setting': self._rate_to_equipment_setting(rate, equipment_params)
                    }
            
            return zones
            
        except Exception as e:
            self.logger.error(f"Error creating equipment zones: {e}")
            return {}
    
    def _rate_to_equipment_setting(self, rate: float, 
                                  equipment_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert irrigation rate to equipment settings"""
        try:
            # Simplified conversion - in practice would be equipment-specific
            return {
                'pressure_bar': min(3.0, max(1.0, rate / 5.0)),  # Example calculation
                'flow_rate_lpm': rate * equipment_params['boom_width_m'] * equipment_params['application_speed_kmh'] / 60,
                'nozzle_selection': 'standard' if rate < 10 else 'high_flow'
            }
            
        except Exception:
            return {'pressure_bar': 2.0, 'flow_rate_lpm': 50.0, 'nozzle_selection': 'standard'}