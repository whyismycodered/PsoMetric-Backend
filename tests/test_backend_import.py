"""Quick import test for the backend package."""
import importlib


def test_import_backend_main():
    m = importlib.import_module("backend.main")
    assert hasattr(m, "app")
