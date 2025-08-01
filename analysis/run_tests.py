#!/usr/bin/env python3
"""
Test runner for conversation analysis tests.

This script runs all tests and generates a coverage report.
"""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run all tests with coverage if available."""
    test_dir = Path(__file__).parent / "tests"
    
    # Check if pytest-cov is available
    try:
        import pytest_cov
        # Run pytest with coverage
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_dir),
            "--cov=embedding_analysis",
            "--cov=run_analysis",
            "--cov-report=html",
            "--cov-report=term",
            "-v"
        ]
        print("Running tests with coverage...")
    except ImportError:
        # Run without coverage
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_dir),
            "-v"
        ]
        print("Running tests without coverage (install pytest-cov for coverage reports)...")
    
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
        if 'pytest_cov' in sys.modules:
            print("Coverage report generated in htmlcov/")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


def run_specific_test(test_name):
    """Run a specific test file or test case."""
    test_dir = Path(__file__).parent / "tests"
    
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir / test_name),
        "-v",
        "-s"  # Show print statements
    ]
    
    print(f"Running specific test: {test_name}")
    subprocess.run(cmd)


def run_unit_tests():
    """Run only unit tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_hierarchical_hypothesis.py",
        "tests/test_paradigm_null_models.py",
        "tests/test_control_analyses.py",
        "-v"
    ]
    
    print("Running unit tests...")
    subprocess.run(cmd)


def run_functional_tests():
    """Run functional tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_hypothesis_data_preparation.py",
        "-v"
    ]
    
    print("Running functional tests...")
    subprocess.run(cmd)


def run_integration_tests():
    """Run integration tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_pipeline_integration.py",
        "tests/test_pipeline.py",
        "-v"
    ]
    
    print("Running integration tests...")
    subprocess.run(cmd)


def run_fast_tests():
    """Run all tests except those marked as slow."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "-m", "not slow"
    ]
    
    print("Running fast tests (excluding slow tests)...")
    subprocess.run(cmd)


def print_usage():
    """Print usage information."""
    print("""
Usage: python run_tests.py [option]

Options:
    (no option)     Run all tests
    unit           Run unit tests only
    functional     Run functional tests only  
    integration    Run integration tests only
    fast           Run all tests except slow ones
    <filename>     Run specific test file
    
Examples:
    python run_tests.py                          # Run all tests
    python run_tests.py unit                     # Run unit tests
    python run_tests.py test_control_analyses.py # Run specific test file
    python run_tests.py fast                     # Run fast tests only
    """)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        option = sys.argv[1]
        if option == "unit":
            run_unit_tests()
        elif option == "functional":
            run_functional_tests()
        elif option == "integration":
            run_integration_tests()
        elif option == "fast":
            run_fast_tests()
        elif option in ["help", "-h", "--help"]:
            print_usage()
        else:
            run_specific_test(option)
    else:
        run_tests()