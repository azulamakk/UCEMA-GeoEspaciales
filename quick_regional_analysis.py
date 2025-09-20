#!/usr/bin/env python3
"""
Quick Regional Analysis for Argentina
Fast analysis of all regions with proper stress calculations and combined maps
"""
import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys
import folium
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from main import WaterStressDetectionSystem

class QuickRegionalAnalysis:
    """Quick analysis of Argentina regions"""
    
    def __init__(self):
        self.system = WaterStressDetectionSystem()
        self.results = {}
        
    async def run_quick_analysis(self):
        """Run quick analysis on all regions"""
        print("🇦🇷 QUICK ARGENTINA WATER STRESS ANALYSIS")
        print("=" * 50)
        
        # Get high-priority small regions
        all_regions = self.system.config['study_areas']
        high_priority_regions = [
            name for name, config in all_regions.items() 
            if config.get('priority') == 'high'
        ][:6]  # Limit to 6 regions for speed
        
        print(f"🗺️  Analyzing {len(high_priority_regions)} regions")
        
        # Use very short date range for speed
        end_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')  # 1 week
        
        print(f"📅 Period: {start_date} to {end_date}")
        print()
        
        results = {}
        successful_count = 0
        
        for i, region_name in enumerate(high_priority_regions, 1):
            region_config = all_regions[region_name]
            print(f"[{i}/{len(high_priority_regions)}] 🌾 {region_config['name']}... ", end="", flush=True)
            
            try:
                # Run quick analysis
                result = await self.system._analyze_single_region(
                    region_name, 'soybean', start_date, end_date
                )
                
                results[region_name] = result
                successful_count += 1
                
                # Extract key info
                alert = result.get('alerts', {}).get('current_alert', {})
                alert_level = alert.get('alert_level', 'unknown')
                alert_score = alert.get('alert_score', 0)
                
                status_emoji = "🟢" if alert_level == 'normal' else "🟡" if alert_level == 'warning' else "🔴"
                print(f"{status_emoji} {alert_level.upper()} ({alert_score:.2f})")
                
            except Exception as e:
                print(f"❌ FAILED ({str(e)[:30]}...)")
                results[region_name] = {'error': str(e)}
        
        print(f"\n✅ Completed: {successful_count}/{len(high_priority_regions)} regions")
        
        # Create comprehensive map
        self._create_comprehensive_map(results, high_priority_regions)
        
        # Save results summary
        self._save_results_summary(results, start_date, end_date)
        
        return results
    
    def _create_comprehensive_map(self, results: dict, regions: list):
        """Create comprehensive interactive map"""
        print("\n🗺️  Creating comprehensive map...")
        
        # Create map centered on Argentina agricultural regions
        m = folium.Map(
            location=[-33.0, -61.0], 
            zoom_start=7,
            tiles='OpenStreetMap'
        )
        
        colors = {
            'normal': 'green',
            'warning': 'orange',
            'critical': 'red',
            'unknown': 'gray'
        }
        
        successful_regions = 0
        
        for region_name in regions:
            if region_name not in results or 'error' in results[region_name]:
                continue
                
            region_config = self.system.config['study_areas'][region_name]
            result = results[region_name]
            
            # Get alert data
            alert = result.get('alerts', {}).get('current_alert', {})
            alert_level = alert.get('alert_level', 'unknown')
            alert_score = alert.get('alert_score', 0)
            
            # Get indicators for detailed popup
            indicators = alert.get('indicators', {})
            
            # Create region boundary
            geometry = region_config['geometry']
            coords = [[lat, lon] for lon, lat in geometry]
            
            color = colors.get(alert_level, 'gray')
            
            # Create detailed popup
            popup_html = f"""
            <div style="width: 300px;">
                <h4>{region_config['name']}</h4>
                <p><b>Alert Level:</b> <span style="color:{color}; font-weight:bold;">{alert_level.upper()}</span></p>
                <p><b>Alert Score:</b> {alert_score:.2f}</p>
                <p><b>Crops:</b> {', '.join(region_config['crops'])}</p>
                
                <h5>Stress Indicators:</h5>
            """
            
            for indicator_name, indicator_data in indicators.items():
                if isinstance(indicator_data, dict) and 'value' in indicator_data:
                    value = indicator_data.get('value', 'N/A')
                    status = indicator_data.get('status', 'unknown')
                    indicator_color = colors.get(status, 'gray')
                    
                    popup_html += f"""
                    <p style="margin: 2px 0;">
                        <b>{indicator_name.upper()}:</b> 
                        <span style="color:{indicator_color};">{value:.3f if isinstance(value, (int, float)) else value}</span>
                        ({status})
                    </p>
                    """
            
            popup_html += "</div>"
            
            # Add polygon to map
            folium.Polygon(
                locations=coords,
                color=color,
                weight=3,
                fillColor=color,
                fillOpacity=0.4,
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"{region_config['name']}: {alert_level.upper()} ({alert_score:.2f})"
            ).add_to(m)
            
            successful_regions += 1
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 220px; height: 160px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px; border-radius: 5px;">
        <h4 style="margin-top: 0;">Water Stress Levels</h4>
        <p style="margin: 3px 0;"><span style="color:green;">●</span> Normal (< 0.4)</p>
        <p style="margin: 3px 0;"><span style="color:orange;">●</span> Warning (0.4 - 0.7)</p>
        <p style="margin: 3px 0;"><span style="color:red;">●</span> Critical (> 0.7)</p>
        <p style="margin: 3px 0;"><span style="color:gray;">●</span> No Data</p>
        <hr style="margin: 5px 0;">
        <p style="margin: 3px 0; font-size: 10px;">Regions analyzed: {successful_regions}</p>
        <p style="margin: 3px 0; font-size: 10px;">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add title
        title_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 50%; transform: translateX(-50%); width: 400px;
                    background-color: white; border:2px solid grey; z-index:9999; 
                    text-align: center; padding: 10px; border-radius: 5px;">
        <h3 style="margin: 0;">Argentina Agricultural Water Stress</h3>
        <p style="margin: 5px 0;">Regional Analysis Dashboard</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save map
        map_file = "outputs/argentina_water_stress_dashboard.html"
        m.save(map_file)
        print(f"💾 Map saved: {map_file}")
        
        return map_file
    
    def _save_results_summary(self, results: dict, start_date: str, end_date: str):
        """Save comprehensive results summary"""
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        # Calculate summary statistics
        alert_levels = []
        alert_scores = []
        
        for result in successful_results.values():
            alert = result.get('alerts', {}).get('current_alert', {})
            alert_levels.append(alert.get('alert_level', 'unknown'))
            alert_scores.append(alert.get('alert_score', 0))
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'period': {'start': start_date, 'end': end_date},
            'total_regions': len(results),
            'successful_analyses': len(successful_results),
            'failed_analyses': len(results) - len(successful_results),
            'national_summary': {
                'alert_distribution': {
                    'normal': alert_levels.count('normal'),
                    'warning': alert_levels.count('warning'),
                    'critical': alert_levels.count('critical')
                },
                'average_alert_score': np.mean(alert_scores) if alert_scores else 0,
                'max_alert_score': max(alert_scores) if alert_scores else 0,
                'regions_at_risk': len([level for level in alert_levels if level in ['warning', 'critical']])
            },
            'regional_details': {
                region: {
                    'alert_level': result.get('alerts', {}).get('current_alert', {}).get('alert_level', 'unknown'),
                    'alert_score': result.get('alerts', {}).get('current_alert', {}).get('alert_score', 0),
                    'indicators': result.get('alerts', {}).get('current_alert', {}).get('indicators', {})
                }
                for region, result in successful_results.items()
            }
        }
        
        # Save summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = f"outputs/argentina_analysis_summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📋 Summary saved: {summary_file}")
        print(f"🎯 Overall status: {len(alert_levels) - alert_levels.count('normal')}/{len(alert_levels)} regions showing stress")

async def main():
    """Main execution"""
    analyzer = QuickRegionalAnalysis()
    
    try:
        results = await analyzer.run_quick_analysis()
        
        print("\n🏁 ANALYSIS COMPLETE!")
        print("🗺️  Open outputs/argentina_water_stress_dashboard.html to see the interactive map")
        print("📊 Check outputs/argentina_analysis_summary_*.json for detailed results")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))