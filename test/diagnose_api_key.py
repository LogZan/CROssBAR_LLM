"""
最终诊断：比较硬编码的 API key 和环境变量中的 API key
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("API KEY COMPARISON")
print("=" * 80)

# 硬编码的 API key (from test_gemini.py)
HARDCODED_KEY = "sk-jhBYth1jQ7d3odY6nnk54Ox3AMffZyMsTCKY7Z4n4MDNZYGJ"

# 从环境变量获取
ENV_KEY = os.getenv("OPENROUTER_API_KEY", "")

print(f"\n1. Hardcoded API Key (from test_gemini.py):")
print(f"   {HARDCODED_KEY[:20]}...{HARDCODED_KEY[-10:]}")

print(f"\n2. Environment Variable (OPENROUTER_API_KEY from .env):")
if ENV_KEY:
    print(f"   {ENV_KEY[:20]}...{ENV_KEY[-10:]}")
else:
    print(f"   ❌ NOT SET!")

print(f"\n3. Are they the same?")
if HARDCODED_KEY == ENV_KEY:
    print(f"   ✅ YES - Keys match!")
else:
    print(f"   ❌ NO - Keys are DIFFERENT!")
    print(f"\n   This is the problem! The backend is using a different API key.")
    print(f"   \n   SOLUTION:")
    print(f"   Update your .env file to use the working API key:")
    print(f'   OPENROUTER_API_KEY="{HARDCODED_KEY}"')

print("\n" + "=" * 80)
print("CHECKING BASE URL")
print("=" * 80)

HARDCODED_BASE = "http://35.220.164.252:3888/v1/"
ENV_BASE = os.getenv("OPENROUTER_API_BASE", "")

print(f"\n1. Hardcoded Base URL (from test_gemini.py):")
print(f"   {HARDCODED_BASE}")

print(f"\n2. Environment Variable (OPENROUTER_API_BASE from .env):")
if ENV_BASE:
    print(f"   {ENV_BASE}")
else:
    print(f"   ❌ NOT SET (will use default)")

print(f"\n3. Are they the same?")
if HARDCODED_BASE.rstrip('/') == ENV_BASE.rstrip('/'):
    print(f"   ✅ YES - URLs match (ignoring trailing slash)!")
else:
    print(f"   ⚠️  Different!")
    if not ENV_BASE.endswith('/') and HARDCODED_BASE.endswith('/'):
        print(f"   Note: Only difference is the trailing slash")
        print(f"   This should be OK, but recommended to add the slash:")
        print(f'   OPENROUTER_API_BASE="{HARDCODED_BASE}"')

print("\n" + "=" * 80)
print("RECOMMENDED .env CONFIGURATION")
print("=" * 80)
print()
print(f'OPENROUTER_API_KEY="{HARDCODED_KEY}"')
print(f'OPENROUTER_API_BASE="{HARDCODED_BASE}"')
print()
