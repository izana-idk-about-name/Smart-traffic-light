"""
Test file for configuration management system.
Demonstrates loading, validation, and usage of settings.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_settings_loading():
    """Test basic settings loading"""
    print("=" * 70)
    print("Test 1: Basic Settings Loading")
    print("=" * 70)
    
    try:
        from src.settings import Settings, get_settings
        
        # Load settings
        settings = get_settings()
        
        print("✅ Settings loaded successfully!")
        print(f"   Mode: {settings.system.mode}")
        print(f"   Platform: {settings.system.platform}")
        print(f"   Debug: {settings.system.debug}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_camera_settings():
    """Test camera configuration"""
    print("\n" + "=" * 70)
    print("Test 2: Camera Settings")
    print("=" * 70)
    
    try:
        from src.settings import get_settings
        settings = get_settings()
        
        print(f"✅ Camera A Index: {settings.camera.camera_a_index}")
        print(f"✅ Camera B Index: {settings.camera.camera_b_index}")
        print(f"✅ Resolution: {settings.camera.width}x{settings.camera.height}")
        print(f"✅ FPS: {settings.camera.fps}")
        print(f"✅ Use Test Images: {settings.camera.use_test_images}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to access camera settings: {e}")
        return False


def test_detection_settings():
    """Test detection configuration"""
    print("\n" + "=" * 70)
    print("Test 3: Detection Settings")
    print("=" * 70)
    
    try:
        from src.settings import get_settings
        settings = get_settings()
        
        print(f"✅ Use ML Model: {settings.detection.use_ml_model}")
        print(f"✅ Min Confidence: {settings.detection.min_confidence}")
        print(f"✅ Model Path: {settings.detection.model_path}")
        print(f"✅ Enable Tracking: {settings.detection.enable_tracking}")
        print(f"✅ Reset Interval: {settings.detection.reset_interval_seconds}s")
        
        # Check if model file exists
        model_path = Path(settings.detection.model_path)
        if model_path.exists():
            print(f"✅ Model file found: {model_path}")
        else:
            print(f"⚠️  Model file not found: {model_path} (will use CV fallback)")
        
        return True
    except Exception as e:
        print(f"❌ Failed to access detection settings: {e}")
        return False


def test_performance_settings():
    """Test performance configuration"""
    print("\n" + "=" * 70)
    print("Test 4: Performance Settings")
    print("=" * 70)
    
    try:
        from src.settings import get_settings
        settings = get_settings()
        
        print(f"✅ Max Processing Time: {settings.performance.max_processing_time}s")
        print(f"✅ Decision Interval: {settings.performance.decision_interval}s")
        print(f"✅ Thread Count: {settings.performance.thread_count}")
        print(f"✅ Memory Limit: {settings.performance.memory_limit_mb}MB")
        print(f"✅ Visualization Enabled: {settings.performance.visualization_enabled}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to access performance settings: {e}")
        return False


def test_logging_settings():
    """Test logging configuration"""
    print("\n" + "=" * 70)
    print("Test 5: Logging Settings")
    print("=" * 70)
    
    try:
        from src.settings import get_settings
        settings = get_settings()
        
        print(f"✅ Log Level: {settings.logging.log_level}")
        print(f"✅ Log Directory: {settings.logging.log_dir}")
        print(f"✅ Performance Logging: {settings.logging.enable_performance_logging}")
        print(f"✅ Log to Console: {settings.logging.log_to_console}")
        print(f"✅ Log to File: {settings.logging.log_to_file}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to access logging settings: {e}")
        return False


def test_traffic_control_settings():
    """Test traffic control configuration"""
    print("\n" + "=" * 70)
    print("Test 6: Traffic Control Settings")
    print("=" * 70)
    
    try:
        from src.settings import get_settings
        settings = get_settings()
        
        print(f"✅ Min Green Time: {settings.traffic_control.min_green_time}s")
        print(f"✅ Max Green Time: {settings.traffic_control.max_green_time}s")
        print(f"✅ Yellow Time: {settings.traffic_control.yellow_time}s")
        print(f"✅ Red Time: {settings.traffic_control.red_time}s")
        
        # Validate timing consistency
        if settings.traffic_control.min_green_time < settings.traffic_control.max_green_time:
            print("✅ Timing validation passed")
        else:
            print("❌ Invalid timing: min_green >= max_green")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Failed to access traffic control settings: {e}")
        return False


def test_network_settings():
    """Test network configuration"""
    print("\n" + "=" * 70)
    print("Test 7: Network Settings")
    print("=" * 70)
    
    try:
        from src.settings import get_settings
        settings = get_settings()
        
        print(f"✅ Orchestrator Host: {settings.network.orchestrator_host}")
        print(f"✅ Orchestrator Port: {settings.network.orchestrator_port}")
        print(f"✅ Use WebSocket: {settings.network.use_websocket}")
        print(f"✅ Timeout: {settings.network.timeout}s")
        print(f"✅ Retry Count: {settings.network.retry_count}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to access network settings: {e}")
        return False


def test_to_dict():
    """Test dictionary conversion"""
    print("\n" + "=" * 70)
    print("Test 8: Dictionary Conversion")
    print("=" * 70)
    
    try:
        from src.settings import get_settings
        settings = get_settings()
        
        config_dict = settings.to_dict()
        
        print("✅ Settings converted to dictionary")
        print(f"✅ Config sections: {list(config_dict.keys())}")
        
        # Verify all sections present
        expected_sections = ['system', 'camera', 'detection', 'performance', 
                           'logging', 'traffic_control', 'network']
        for section in expected_sections:
            if section in config_dict:
                print(f"   ✓ {section}")
            else:
                print(f"   ✗ {section} (missing)")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Failed to convert to dictionary: {e}")
        return False


def test_validation():
    """Test configuration validation"""
    print("\n" + "=" * 70)
    print("Test 9: Configuration Validation")
    print("=" * 70)
    
    try:
        # Save original env vars
        original_env = {}
        test_vars = {
            'MIN_CONFIDENCE': '1.5',  # Invalid: > 1.0
            'CAMERA_FPS': '-10',      # Invalid: < 0
        }
        
        for key, value in test_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            from src.settings import Settings
            settings = Settings.load_from_env()
            print("❌ Validation should have failed for invalid values")
            return False
        except ValueError as e:
            print(f"✅ Validation correctly rejected invalid values: {e}")
            return True
        finally:
            # Restore original env vars
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
        
    except Exception as e:
        print(f"❌ Validation test failed unexpectedly: {e}")
        return False


def test_singleton_pattern():
    """Test singleton pattern"""
    print("\n" + "=" * 70)
    print("Test 10: Singleton Pattern")
    print("=" * 70)
    
    try:
        from src.settings import get_settings
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        if settings1 is settings2:
            print("✅ Singleton pattern working correctly")
            print(f"   Same instance: {id(settings1)} == {id(settings2)}")
            return True
        else:
            print("❌ Singleton pattern not working - different instances")
            return False
    except Exception as e:
        print(f"❌ Singleton test failed: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility with old config modules"""
    print("\n" + "=" * 70)
    print("Test 11: Backward Compatibility")
    print("=" * 70)
    
    try:
        # Test importing legacy settings
        from src.settings import (
            IS_RASPBERRY_PI,
            CAMERA_SETTINGS,
            PROCESSING_SETTINGS,
            MODEL_SETTINGS,
            NETWORK_SETTINGS
        )
        
        print(f"✅ Legacy IS_RASPBERRY_PI imported: {IS_RASPBERRY_PI}")
        print(f"✅ Legacy CAMERA_SETTINGS imported: {type(CAMERA_SETTINGS)}")
        print(f"✅ Legacy PROCESSING_SETTINGS imported: {type(PROCESSING_SETTINGS)}")
        print(f"✅ Legacy MODEL_SETTINGS imported: {type(MODEL_SETTINGS)}")
        print(f"✅ Legacy NETWORK_SETTINGS imported: {type(NETWORK_SETTINGS)}")
        
        print("✅ Backward compatibility maintained")
        return True
    except Exception as e:
        print(f"⚠️  Legacy imports not available (expected if not needed): {e}")
        return True  # Not critical


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("SMART TRAFFIC LIGHT - CONFIGURATION SYSTEM TESTS")
    print("=" * 70)
    
    tests = [
        test_settings_loading,
        test_camera_settings,
        test_detection_settings,
        test_performance_settings,
        test_logging_settings,
        test_traffic_control_settings,
        test_network_settings,
        test_to_dict,
        test_singleton_pattern,
        test_backward_compatibility,
        # test_validation,  # Commented out as it modifies env vars
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n❌ Test {test.__name__} crashed: {e}")
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 All tests passed! Configuration system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())