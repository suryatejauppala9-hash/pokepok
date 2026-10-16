"""
Startup script for PokéGuru — AI Pokémon Mentor
Serves both backend API and frontend from one place
"""
import subprocess
import sys
import os
import webbrowser
import time
from pathlib import Path

def main():
    backend_dir = Path(__file__).parent / "backend"
    
    print("=" * 60)
    print("  🎓 PokéGuru — AI Pokémon Mentor")
    print("  Professor Oak speaks your language!")
    print("=" * 60)
    print()
    
    # Check for optional Groq key
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        print(f"  ✅ Groq API key found — LLM reasoning enabled")
    else:
        print("  ℹ️  No GROQ_API_KEY set — using rule-based advice")
        print("     Get a free key at https://console.groq.com")
    
    print()
    print("  🚀 Starting server at http://localhost:8000")
    print("  📖 API docs at    http://localhost:8000/docs")
    print()
    print("  Press Ctrl+C to stop")
    print("-" * 60)
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:8000")
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run uvicorn
    os.chdir(backend_dir)
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ])

if __name__ == "__main__":
    main()
