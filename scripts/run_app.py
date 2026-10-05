"""Launch the BANKING77 Streamlit Dashboard."""
import subprocess
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print(f"Launching Streamlit app: {app_path}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.headless", "true",
    ])


if __name__ == '__main__':
    main()
