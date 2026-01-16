# Remove the proxy unset if you need proxy to access Gemini API
# env -u ALL_PROXY -u HTTPS_PROXY -u HTTP_PROXY -u all_proxy -u https_proxy -u http_proxy poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Use this instead to keep proxy settings:
poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000