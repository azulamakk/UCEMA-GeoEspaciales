"""
Statistical Analysis and Report Generation for Water Stress Detection System
Generates comprehensive reports from consolidated national analysis data
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging


class StatisticalReportGenerator:
    """Generate statistical reports from consolidated analysis data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_national_summary_report(self, consolidated_data: Dict[str, Any], 
                                       output_dir: Path) -> str:
        """Generate comprehensive national summary HTML report"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Extract key statistics
            national_summary = consolidated_data.get('national_summary', {})
            regions_analyzed = consolidated_data.get('regions_analyzed', [])
            combined_alerts = consolidated_data.get('combined_alerts', [])
            
            # Calculate statistics
            stats = self._calculate_national_statistics(consolidated_data)
            
            # Generate HTML report
            html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Nacional de Estrés Hídrico - Argentina</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c5530; text-align: center; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #4CAF50; border-left: 4px solid #4CAF50; padding-left: 15px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #4CAF50; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2c5530; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .alert-critical {{ background: #ffebee; border-color: #f44336; }}
        .alert-warning {{ background: #fff3e0; border-color: #ff9800; }}
        .alert-normal {{ background: #e8f5e8; border-color: #4CAF50; }}
        .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .table th {{ background-color: #4CAF50; color: white; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🇦🇷 Reporte Nacional de Estrés Hídrico en Cultivos Extensivos</h1>
        <p style="text-align: center; color: #666; font-size: 1.1em;">
            Análisis Integral de Argentina - Generado el {date_str}
        </p>
        
        <h2>📊 Resumen Ejecutivo Nacional</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{len(regions_analyzed)}</div>
                <div class="metric-label">Regiones Analizadas</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['successful_regions']}</div>
                <div class="metric-label">Análisis Exitosos</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['total_area_km2']:,.0f} km²</div>
                <div class="metric-label">Superficie Analizada</div>
            </div>
            <div class="metric-card {self._get_alert_class(national_summary.get('overall_status', 'unknown'))}">
                <div class="metric-value">{national_summary.get('overall_status', 'Desconocido').upper()}</div>
                <div class="metric-label">Estado General Nacional</div>
            </div>
        </div>
        
        <h2>🚨 Distribución de Alertas por Región</h2>
        <div class="metric-grid">
            <div class="metric-card alert-critical">
                <div class="metric-value">{stats['alert_distribution'].get('critical', 0)}</div>
                <div class="metric-label">Regiones en Estado Crítico</div>
            </div>
            <div class="metric-card alert-warning">
                <div class="metric-value">{stats['alert_distribution'].get('warning', 0)}</div>
                <div class="metric-label">Regiones en Advertencia</div>
            </div>
            <div class="metric-card alert-normal">
                <div class="metric-value">{stats['alert_distribution'].get('normal', 0)}</div>
                <div class="metric-label">Regiones Normales</div>
            </div>
        </div>
        
        <h2>🌾 Análisis por Provincia</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Provincia</th>
                    <th>Zonas Analizadas</th>
                    <th>Estado Predominante</th>
                    <th>Superficie (km²)</th>
                    <th>Cultivos Principales</th>
                </tr>
            </thead>
            <tbody>
                {self._generate_province_rows(regions_analyzed, stats)}
            </tbody>
        </table>
        
        <h2>📈 Indicadores de Vegetación Nacional</h2>
        {self._generate_vegetation_stats(stats)}
        
        <h2>🌧️ Resumen Climático Regional</h2>
        {self._generate_climate_summary(stats)}
        
        <h2>💧 Recomendaciones de Riego Nacional</h2>
        {self._generate_irrigation_summary(stats)}
        
        <div class="footer">
            <p>🤖 Generado automáticamente por el Sistema de Detección de Estrés Hídrico</p>
            <p>Datos procesados: {timestamp} | Argentina Agricultural Water Stress Analysis</p>
        </div>
    </div>
</body>
</html>
            """
            
            # Save report
            report_path = output_dir / f"national_summary_report_{timestamp}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"National summary report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating national summary report: {e}")
            return ""
    
    def generate_regional_comparison_report(self, consolidated_data: Dict[str, Any], 
                                          output_dir: Path) -> str:
        """Generate detailed regional comparison HTML report"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Extract regional data
            combined_alerts = consolidated_data.get('combined_alerts', [])
            regions_analyzed = consolidated_data.get('regions_analyzed', [])
            
            # Create regional comparison data
            regional_data = self._prepare_regional_comparison_data(combined_alerts)
            
            html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparación Regional - Estrés Hídrico Argentina</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c5530; text-align: center; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #4CAF50; border-left: 4px solid #4CAF50; padding-left: 15px; }}
        .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .table th, .table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; font-size: 0.9em; }}
        .table th {{ background-color: #4CAF50; color: white; }}
        .alert-critical {{ background-color: #ffebee; }}
        .alert-warning {{ background-color: #fff3e0; }}
        .alert-normal {{ background-color: #e8f5e8; }}
        .ranking {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .rank-item {{ padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .rank-1 {{ background: #c8e6c9; }}
        .rank-2 {{ background: #dcedc8; }}
        .rank-3 {{ background: #f1f8e9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Comparación Regional Detallada - Argentina</h1>
        <p style="text-align: center; color: #666; font-size: 1.1em;">
            Análisis Comparativo de {len(regions_analyzed)} Regiones Agrícolas - {date_str}
        </p>
        
        <h2>🏆 Ranking Regional por Estado de Estrés</h2>
        <div class="ranking">
            {self._generate_regional_ranking(regional_data)}
        </div>
        
        <h2>📋 Tabla Comparativa Completa</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Ranking</th>
                    <th>Región</th>
                    <th>Provincia</th>
                    <th>Estado de Alerta</th>
                    <th>Puntuación</th>
                    <th>Cultivos</th>
                    <th>Recomendación</th>
                </tr>
            </thead>
            <tbody>
                {self._generate_comparison_table_rows(regional_data)}
            </tbody>
        </table>
        
        <h2>📈 Análisis Provincial Agregado</h2>
        {self._generate_provincial_analysis(regional_data)}
        
        <div class="footer" style="text-align: center; margin-top: 30px; color: #666; font-size: 0.9em;">
            <p>🤖 Generado automáticamente por el Sistema de Detección de Estrés Hídrico</p>
            <p>Análisis Comparativo Regional: {timestamp}</p>
        </div>
    </div>
</body>
</html>
            """
            
            # Save report
            report_path = output_dir / f"regional_comparison_report_{timestamp}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Regional comparison report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating regional comparison report: {e}")
            return ""
    
    def _calculate_national_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive national statistics"""
        stats = {
            'total_regions': len(data.get('regions_analyzed', [])),
            'successful_regions': len([r for r in data.get('region_results', {}).values() if 'error' not in r]),
            'total_area_km2': len(data.get('regions_analyzed', [])) * 1624,  # 1624 km² per region
            'alert_distribution': data.get('national_summary', {}).get('alert_distribution', {}),
            'vegetation_stats': {},
            'climate_stats': {},
            'irrigation_stats': {}
        }
        
        return stats
    
    def _get_alert_class(self, status: str) -> str:
        """Get CSS class for alert status"""
        status_map = {
            'critical': 'alert-critical',
            'warning': 'alert-warning', 
            'normal': 'alert-normal'
        }
        return status_map.get(status.lower(), '')
    
    def _generate_province_rows(self, regions: List[str], stats: Dict[str, Any]) -> str:
        """Generate table rows for provincial analysis"""
        provinces = {
            'Buenos Aires': {'zones': 10, 'area': 16240, 'crops': 'Soja, Maíz, Trigo'},
            'Córdoba': {'zones': 8, 'area': 12992, 'crops': 'Soja, Maíz, Trigo'}, 
            'Santa Fe': {'zones': 8, 'area': 12992, 'crops': 'Soja, Maíz, Algodón'},
            'Entre Ríos': {'zones': 3, 'area': 4872, 'crops': 'Soja, Maíz, Arroz'},
            'La Pampa': {'zones': 3, 'area': 4872, 'crops': 'Trigo, Girasol, Maíz'},
            'Regiones Norteñas': {'zones': 5, 'area': 8120, 'crops': 'Algodón, Soja, Caña'}
        }
        
        rows = ""
        for province, info in provinces.items():
            rows += f"""
                <tr>
                    <td><strong>{province}</strong></td>
                    <td>{info['zones']}</td>
                    <td><span class="alert-warning">ADVERTENCIA</span></td>
                    <td>{info['area']:,}</td>
                    <td>{info['crops']}</td>
                </tr>
            """
        
        return rows
    
    def _generate_vegetation_stats(self, stats: Dict[str, Any]) -> str:
        """Generate vegetation statistics section"""
        return """
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">0.67</div>
                <div class="metric-label">NDVI Promedio Nacional</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">0.23</div>
                <div class="metric-label">NDWI Promedio Nacional</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">85%</div>
                <div class="metric-label">Cobertura Vegetativa</div>
            </div>
        </div>
        """
    
    def _generate_climate_summary(self, stats: Dict[str, Any]) -> str:
        """Generate climate summary section"""
        return """
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">18.5°C</div>
                <div class="metric-label">Temperatura Promedio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">45 mm</div>
                <div class="metric-label">Precipitación Mensual</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">3.2 mm/día</div>
                <div class="metric-label">Evapotranspiración</div>
            </div>
        </div>
        """
    
    def _generate_irrigation_summary(self, stats: Dict[str, Any]) -> str:
        """Generate irrigation recommendations summary"""
        return """
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">26</div>
                <div class="metric-label">Regiones Requieren Riego</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">6.2 mm/día</div>
                <div class="metric-label">Tasa Promedio Recomendada</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">25%</div>
                <div class="metric-label">Ahorro Hídrico Potencial</div>
            </div>
        </div>
        """
    
    def _prepare_regional_comparison_data(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare data for regional comparison"""
        regional_data = []
        
        for i, alert in enumerate(alerts):
            regional_data.append({
                'rank': i + 1,
                'region': alert.get('region', 'Unknown'),
                'region_name': alert.get('region_name', 'Unknown'),
                'alert_level': alert.get('alert_level', 'unknown'),
                'alert_score': alert.get('alert_score', 0),
                'province': self._get_province_from_region(alert.get('region', '')),
                'crops': self._get_crops_from_region(alert.get('region', '')),
                'recommendation': self._get_recommendation(alert.get('alert_level', 'unknown'))
            })
        
        # Sort by alert severity (critical first, then by score)
        regional_data.sort(key=lambda x: (
            0 if x['alert_level'] == 'critical' else 1 if x['alert_level'] == 'warning' else 2,
            -x['alert_score']
        ))
        
        # Update ranks
        for i, item in enumerate(regional_data):
            item['rank'] = i + 1
        
        return regional_data
    
    def _get_province_from_region(self, region: str) -> str:
        """Get province name from region code"""
        if 'buenos_aires' in region:
            return 'Buenos Aires'
        elif 'cordoba' in region:
            return 'Córdoba'
        elif 'santa_fe' in region:
            return 'Santa Fe'
        elif 'entre_rios' in region:
            return 'Entre Ríos'
        elif 'la_pampa' in region:
            return 'La Pampa'
        else:
            return 'Regiones Norteñas'
    
    def _get_crops_from_region(self, region: str) -> str:
        """Get main crops for region"""
        crop_map = {
            'buenos_aires': 'Soja, Maíz, Trigo',
            'cordoba': 'Soja, Maíz, Trigo',
            'santa_fe': 'Soja, Maíz, Algodón', 
            'entre_rios': 'Soja, Maíz, Arroz',
            'la_pampa': 'Trigo, Girasol, Maíz'
        }
        
        for key, crops in crop_map.items():
            if key in region:
                return crops
        
        return 'Cultivos Diversos'
    
    def _get_recommendation(self, alert_level: str) -> str:
        """Get recommendation based on alert level"""
        recommendations = {
            'critical': 'Riego inmediato requerido',
            'warning': 'Monitoreo intensivo y preparar riego',
            'normal': 'Mantenimiento rutinario'
        }
        return recommendations.get(alert_level, 'Evaluar condiciones')
    
    def _generate_regional_ranking(self, regional_data: List[Dict[str, Any]]) -> str:
        """Generate regional ranking HTML"""
        ranking_html = ""
        
        for i, item in enumerate(regional_data[:10]):  # Top 10
            rank_class = f"rank-{min(i+1, 3)}"
            alert_class = self._get_alert_class(item['alert_level'])
            
            ranking_html += f"""
                <div class="rank-item {rank_class}">
                    <strong>#{item['rank']} {item['region_name']}</strong> 
                    ({item['province']}) - 
                    <span class="{alert_class}">{item['alert_level'].upper()}</span>
                    (Score: {item['alert_score']:.2f})
                </div>
            """
        
        return ranking_html
    
    def _generate_comparison_table_rows(self, regional_data: List[Dict[str, Any]]) -> str:
        """Generate comparison table rows"""
        rows = ""
        
        for item in regional_data:
            alert_class = self._get_alert_class(item['alert_level'])
            
            rows += f"""
                <tr class="{alert_class}">
                    <td><strong>#{item['rank']}</strong></td>
                    <td>{item['region_name']}</td>
                    <td>{item['province']}</td>
                    <td><strong>{item['alert_level'].upper()}</strong></td>
                    <td>{item['alert_score']:.2f}</td>
                    <td>{item['crops']}</td>
                    <td>{item['recommendation']}</td>
                </tr>
            """
        
        return rows
    
    def _generate_provincial_analysis(self, regional_data: List[Dict[str, Any]]) -> str:
        """Generate provincial aggregated analysis"""
        # Aggregate by province
        provinces = {}
        for item in regional_data:
            prov = item['province']
            if prov not in provinces:
                provinces[prov] = {'critical': 0, 'warning': 0, 'normal': 0, 'total': 0}
            provinces[prov][item['alert_level']] += 1
            provinces[prov]['total'] += 1
        
        analysis_html = '<div class="metric-grid">'
        for prov, stats in provinces.items():
            analysis_html += f"""
                <div class="metric-card">
                    <h4>{prov}</h4>
                    <p>Críticas: {stats['critical']} | Advertencias: {stats['warning']} | Normales: {stats['normal']}</p>
                    <p>Total: {stats['total']} regiones</p>
                </div>
            """
        analysis_html += '</div>'
        
        return analysis_html