#!/bin/sh
set -e

# Decode Google credentials from environment variable if provided
if [ -n "$GOOGLE_CREDENTIALS_JSON" ]; then
    echo "$GOOGLE_CREDENTIALS_JSON" | base64 -d > /app/google-credentials.json
    export GOOGLE_APPLICATION_CREDENTIALS=/app/google-credentials.json
fi

exec "$@"
