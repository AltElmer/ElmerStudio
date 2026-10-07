"""Frozen-application entry point (PyInstaller)."""
import sys

from elmerstudio.app import main

if __name__ == "__main__":
    sys.exit(main())
