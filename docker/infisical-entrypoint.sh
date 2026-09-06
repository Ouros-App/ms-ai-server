#!/bin/sh
set -eu

: "${INFISICAL_TOKEN:?INFISICAL_TOKEN is required}"
: "${INFISICAL_PROJECT_ID:?INFISICAL_PROJECT_ID is required}"

exec infisical run \
  --projectId="$INFISICAL_PROJECT_ID" \
  --env="${INFISICAL_ENV:-prod}" \
  --path="${INFISICAL_PATH:-/ms-ai-server}" \
  -- "$@"
