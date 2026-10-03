"""
Root Web Application Entrypoint for Live Interactive Visualizer.
"""

import os
import sys

# Ensure src is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.visualization.web_dashboard import run_dashboard_server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run_dashboard_server(host="0.0.0.0", port=port)
