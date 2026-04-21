from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    # Try reading from config.py if .env isn't loaded correctly
    try:
        from config import settings
        api_key = settings.google_api_key
    except:
        pass

if not api_key:
    print("No API key found")
    exit(1)

client = genai.Client(api_key=api_key)

print("Listing available models for generateContent:")
try:
    models = client.models.list()
    for m in models:
        # Check if generateContent is supported
        if "generateContent" in m.supported_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
