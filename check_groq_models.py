import os
from groq import Groq

# Replace with your actual Groq API key, or ensure it's in your environment
client = Groq(api_key="your_api_key_here")

print("🔍 Fetching available models from Groq...\n")
try:
    models = client.models.list()
    for model in models.data:
        print(f"✅ Available: {model.id}")
except Exception as e:
    print(f"❌ Error: {e}")