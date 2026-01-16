import os
import sys
from dotenv import load_dotenv

# Load .env
load_dotenv()

print("=" * 80)
print("ENVIRONMENT VARIABLES DIAGNOSTIC")
print("=" * 80)

vars_to_check = [
    'OPENROUTER_API_KEY',
    'OPENROUTER_API_BASE',
    'OPENAI_API_KEY',
    'GEMINI_API_KEY',
    'NEO4J_URI',
    'NEO4J_USERNAME',
    'MY_NEO4J_PASSWORD',
    'NEO4J_DATABASE_NAME',
]

for var in vars_to_check:
    value = os.getenv(var)
    if value:
        if 'KEY' in var or 'PASSWORD' in var:
            # Mask sensitive data
            masked = f"{value[:10]}...{value[-4:]}" if len(value) > 14 else "***"
            print(f"✅ {var:30s} = {masked}")
        else:
            print(f"✅ {var:30s} = {value}")
    else:
        print(f"❌ {var:30s} = NOT SET")

print("\n" + "=" * 80)
print("CHECKING URL FORMATTING")
print("=" * 80)

base_url = os.getenv('OPENROUTER_API_BASE', 'https://openrouter.ai/api/v1')
print(f"OPENROUTER_API_BASE: {base_url}")

# Check for common issues
if not base_url.startswith('http'):
    print("⚠️  WARNING: URL doesn't start with http/https")
if base_url.endswith('/'):
    print("✅ URL ends with / (good for compatibility)")
else:
    print("⚠️  URL doesn't end with / (might cause issues)")
    
# Recommend the correct format
print(f"\nRecommended format: http://35.220.164.252:3888/v1/")
print(f"Current format:     {base_url}")

print("\n" + "=" * 80)
print("CHECKING IF .env FILE EXISTS")
print("=" * 80)

env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_path):
    print(f"✅ .env file found at: {env_path}")
    print(f"   File size: {os.path.getsize(env_path)} bytes")
    print(f"   Last modified: {os.path.getmtime(env_path)}")
else:
    print(f"❌ .env file NOT found at: {env_path}")

print("\n" + "=" * 80)
print("INSTRUCTIONS TO FIX")
print("=" * 80)
print("Add these lines to your .env file:")
print()
print('OPENROUTER_API_KEY="sk-jhBYth1jQ7d3odY6nnk54Ox3AMffZyMsTCKY7Z4n4MDNZYGJ"')
print('OPENROUTER_API_BASE="http://35.220.164.252:3888/v1/"')
print()
print("Then restart the backend service.")
