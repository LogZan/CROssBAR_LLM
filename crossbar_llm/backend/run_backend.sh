# Remove the proxy unset if you need proxy to access Gemini API
# env -u ALL_PROXY -u HTTPS_PROXY -u HTTP_PROXY -u all_proxy -u https_proxy -u http_proxy poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Use poetry when available, fall back to plain uvicorn otherwise
if command -v poetry &> /dev/null && poetry env info --path &> /dev/null; then
    poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000
else
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
fi