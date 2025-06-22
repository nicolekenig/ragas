"""Script to run all tests with detailed reporting."""

import subprocess
import sys
import os
# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests():
    """Run all tests with detailed output."""
    print("🧪 Running Cohere AI Assistant Tests")
    print("=" * 50)

    # # Check if required packages are installed
    # required_packages = ['pytest', 'ragas', 'pandas', 'datasets']
    # missing_packages = []
    #
    # for package in required_packages:
    #     try:
    #         __import__(package)
    #     except ImportError:
    #         missing_packages.append(package)
    #
    # if missing_packages:
    #     print(f"❌ Missing packages: {', '.join(missing_packages)}")
    #     print("Install with: pip install -r requirements_test.txt")
    #     return False

    # Run tests
    test_commands = [
        # Basic functionality tests
        ["pytest", "tests/test_assistant_basic.py", "-v", "--tb=short"],

        # RAGAS evaluation tests
        ["pytest", "tests/test_ragas_evaluation.py", "-v", "--tb=short", "-s"],

        # Performance tests
        ["pytest", "tests/test_performance.py", "-v", "--tb=short"],

        # Integration tests
        ["pytest", "tests/test_integration.py", "-v", "--tb=short"],

        # Generate coverage report
        ["pytest", "--cov=assistant", "--cov-report=html", "--cov-report=term"]
    ]

    all_passed = True

    for i, cmd in enumerate(test_commands, 1):
        print(f"\n🔍 Running Test Suite {i}/{len(test_commands)}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 30)

        try:
            result = subprocess.run(cmd, capture_output=False, check=True)
            print(f"✅ Test Suite {i} PASSED")
        except subprocess.CalledProcessError as e:
            print(f"❌ Test Suite {i} FAILED with code {e.returncode}")
            all_passed = False
        except FileNotFoundError:
            print(f"❌ Command not found: {cmd[0]}")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests PASSED!")
        print("📊 Check htmlcov/index.html for coverage report")
    else:
        print("❌ Some tests FAILED!")

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)