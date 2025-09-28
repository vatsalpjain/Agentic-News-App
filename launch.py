import asyncio
import logging
from pathlib import Path
from src.main import main
from dotenv import load_dotenv

def setup_environment():
    """Setup required directories and environment"""
    # Create necessary directories
    directories = [
        Path("data/all_files"),
        Path("data/vector_store"),
        Path("logs")
    ]
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

if __name__ == "__main__":
    # Setup environment
    setup_environment()
    
    # Load environment variables
    load_dotenv()
    
    # Run the application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error: {e}")