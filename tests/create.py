# setup_tests.py
import os
import pathlib

def create_test_structure():
    """Create the test directory structure"""
    
    # Base directories
    directories = [
        'test_fixtures/callables/basic',
        'test_fixtures/callables/async',
        'test_fixtures/callables/imports',
        'test_fixtures/callables/models',
        'test_fixtures/data',
    ]

    # Test files
    files = [
        '__init__.py',
        'conftest.py',
        'test_registration.py',
        'test_execution.py',
        'test_schemas.py',
        'test_async.py',
        'test_imports.py',
        'test_logging.py',
        'test_fixtures/callables/basic/__init__.py',
        'test_fixtures/callables/async/__init__.py',
        'test_fixtures/callables/imports/__init__.py',
        'test_fixtures/callables/models/__init__.py',
    ]

    # Create directories
    for directory in directories:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

    # Create files
    for file in files:
        path = pathlib.Path(file)
        if not path.exists():
            path.touch()
            print(f"Created file: {file}")

if __name__ == "__main__":
    create_test_structure()