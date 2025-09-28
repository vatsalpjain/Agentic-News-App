import asyncio
import os
from src.main import main
from dotenv import load_dotenv

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Ensure required API keys are present
    required_keys = ["GROQ_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {missing_keys}")
    
    # Run the application
    asyncio.run(main())