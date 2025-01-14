# run_tests.py
import pytest
import sys

def run_specific_test(test_name=None, verbose=True, pdb=False):
    """
    Run specific test(s) with detailed output
    
    Args:
        test_name: Optional name of specific test to run
        verbose: Whether to show detailed output
        pdb: Whether to drop into pdb on failure
    """
    args = [
        "-v" if verbose else "",  # Verbose output
        "--pdb" if pdb else "",   # Drop into pdb on failures
        "-s",                     # Don't capture stdout
        "--log-cli-level=DEBUG",  # Show debug logs
    ]
    
    if test_name:
        # Can be test file or specific test
        args.append(f"test_schemas.py::{test_name}" if "::" in test_name else test_name)
    else:
        args.append("test_schemas.py")
    
    # Remove empty args
    args = [arg for arg in args if arg]
    
    return pytest.main(args)

if __name__ == "__main__":
    test_name = sys.argv[1] if len(sys.argv) > 1 else None
    run_specific_test(test_name, verbose=True, pdb=True)