# Remove the proxy unset if you need proxy to access Gemini API
# env -u ALL_PROXY -u HTTPS_PROXY -u HTTP_PROXY -u all_proxy -u https_proxy -u http_proxy uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Use poetry when available, fall back to plain uvicorn otherwise.
# Pass --reload only during local development (CROSSBAR_ENV != production).
RELOAD_FLAG=""
if [ "${CROSSBAR_ENV}" != "production" ]; then
    RELOAD_FLAG="--reload"
fi

if command -v poetry &> /dev/null; then
    poetry run uvicorn main:app $RELOAD_FLAG --host 0.0.0.0 --port 8000
else
    uvicorn main:app $RELOAD_FLAG --host 0.0.0.0 --port 8000
fi