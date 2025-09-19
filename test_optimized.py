#!/usr/bin/env python3
"""
Test script with optimized parameters for large areas
"""
import sys
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from main import WaterStressDetectionSystem

def setup_logging():
    """Setup logging for test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_optimized.log'),
            logging.StreamHandler()
        ]
    )

async def test_small_area():
    """Test with small area first"""
    print("🧪 Testing with small area...")
    
    system = WaterStressDetectionSystem()
    
    # Use shorter time period to reduce timeout risk
    end_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    try:
        results = await system.run_full_analysis(
            study_area='test_area_small',
            crop_type='soybean',
            start_date=start_date,
            end_date=end_date,
            multi_region=False
        )
        
        print("✅ Small area test successful!")
        print(f"Results saved to: {results.get('output_path', 'Unknown')}")
        
        if 'alerts' in results and 'current_alert' in results['alerts']:
            alert = results['alerts']['current_alert']
            print(f"Alert level: {alert['alert_level']}")
            print(f"Alert score: {alert['alert_score']:.2f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Small area test failed: {e}")
        return False

async def test_medium_area():
    """Test with medium area"""
    print("\n🧪 Testing with medium area (Buenos Aires North)...")
    
    system = WaterStressDetectionSystem()
    
    # Use even shorter time period for medium area
    end_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=21)).strftime('%Y-%m-%d')
    
    try:
        results = await system.run_full_analysis(
            study_area='buenos_aires_north',
            crop_type='soybean',
            start_date=start_date,
            end_date=end_date,
            multi_region=False
        )
        
        print("✅ Medium area test successful!")
        print(f"Results saved to: {results.get('output_path', 'Unknown')}")
        
        if 'alerts' in results and 'current_alert' in results['alerts']:
            alert = results['alerts']['current_alert']
            print(f"Alert level: {alert['alert_level']}")
            print(f"Alert score: {alert['alert_score']:.2f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Medium area test failed: {e}")
        return False

async def test_conservative_large_area():
    """Test with large area using conservative parameters"""
    print("\n🧪 Testing with large area (Pampas Central) - Conservative mode...")
    
    system = WaterStressDetectionSystem()
    
    # Use minimal time period for large area
    end_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')  # Only 7 days of data
    
    try:
        results = await system.run_full_analysis(
            study_area='pampas_central',
            crop_type='soybean',
            start_date=start_date,
            end_date=end_date,
            multi_region=False
        )
        
        print("✅ Large area test successful!")
        print(f"Results saved to: {results.get('output_path', 'Unknown')}")
        
        if 'alerts' in results and 'current_alert' in results['alerts']:
            alert = results['alerts']['current_alert']
            print(f"Alert level: {alert['alert_level']}")
            print(f"Alert score: {alert['alert_score']:.2f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Large area test failed: {e}")
        return False

async def main():
    """Main test function"""
    setup_logging()
    
    print("🚀 Starting optimized water stress detection tests...\n")
    print("=" * 60)
    
    # Test progression: Small → Medium → Large
    tests = [
        ("Small Area", test_small_area),
        ("Medium Area", test_medium_area), 
        ("Large Area (Conservative)", test_conservative_large_area)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📍 Running {test_name} test...")
        try:
            success = await test_func()
            results[test_name] = success
            
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:25} {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! System is working correctly.")
    elif passed_tests > 0:
        print("⚠️  Some tests passed. System partially functional.")
    else:
        print("🚨 All tests failed. Check configuration and connectivity.")
    
    return passed_tests > 0

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)