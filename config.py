import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_config():
    """Returns a dictionary of configuration values."""
    config = {
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "day22-lab"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    }
    return config

def verify_config():
    """Verifies that all required configuration values are present."""
    config = get_config()
    required_keys = ["LANGCHAIN_API_KEY", "OPENAI_API_KEY"]
    
    missing_keys = [key for key in required_keys if not config[key] or "your-" in config[key]]
    
    if not missing_keys:
        print("[SUCCESS] Config loaded successfully")
        print(f"   LangSmith project : {config['LANGCHAIN_PROJECT']}")
        print(f"   OpenAI endpoint   : {config['OPENAI_API_BASE']}")
        print(f"   Default LLM model : {config['DEFAULT_MODEL']}")
        print(f"   Embedding model   : {config['EMBEDDING_MODEL']}")
    else:
        print("[ERROR] Config validation failed")
        for key in missing_keys:
            print(f"   Missing or placeholder value for: {key}")
        print("\nPlease update your .env file with actual API keys.")

if __name__ == "__main__":
    verify_config()
