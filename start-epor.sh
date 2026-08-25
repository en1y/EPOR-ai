#!/usr/bin/env bash

set -Eeuo pipefail
set +m

readonly SCRIPT_NAME="${0##*/}"
readonly PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly API_URL="http://127.0.0.1:8742"
readonly UI_URL="http://127.0.0.1:5173"

declare -a SERVICE_PIDS=()

usage() {
  cat <<EOF
Usage: ./${SCRIPT_NAME}

Start the EPOR API, worker, and browser console together.

  -h, --help
           Show this help message.

Press Ctrl+C to stop every service.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  local command_name="$1"
  local install_hint="$2"

  command -v "${command_name}" >/dev/null 2>&1 || fail "${install_hint}"
}

process_group_is_running() {
  kill -0 -- "-$1" 2>/dev/null
}

service_is_running() {
  process_group_is_running "$1" || kill -0 "$1" 2>/dev/null
}

signal_service() {
  local signal_name="$1"
  local pid="$2"

  kill -s "${signal_name}" -- "-${pid}" 2>/dev/null || \
    kill -s "${signal_name}" "${pid}" 2>/dev/null || true
}

cleanup() {
  local exit_code="${1:-0}"
  local pid
  local attempt
  local any_running

  trap - EXIT INT TERM HUP

  if ((${#SERVICE_PIDS[@]} > 0)); then
    printf '\nStopping EPOR services...\n'
    for pid in "${SERVICE_PIDS[@]}"; do
      signal_service TERM "${pid}"
    done

    for ((attempt = 0; attempt < 50; attempt++)); do
      any_running=false
      for pid in "${SERVICE_PIDS[@]}"; do
        if service_is_running "${pid}"; then
          any_running=true
          break
        fi
      done
      if [[ "${any_running}" == false ]]; then
        break
      fi
      sleep 0.1
    done

    for pid in "${SERVICE_PIDS[@]}"; do
      if service_is_running "${pid}"; then
        signal_service KILL "${pid}"
      fi
      wait "${pid}" 2>/dev/null || true
    done
    printf 'EPOR stopped.\n'
  fi

  exit "${exit_code}"
}

start_service() {
  local service_name="$1"
  local attempt
  shift

  setsid "$@" &
  local pid=$!
  SERVICE_PIDS+=("${pid}")

  # setsid establishes the new group just after the background PID is returned.
  # Wait out that small race so later negative-PGID signals always hit descendants.
  for ((attempt = 0; attempt < 100; attempt++)); do
    if process_group_is_running "${pid}" || ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 0.01
  done
  printf 'Started %-6s (pid %s)\n' "${service_name}" "${pid}"
}

api_is_ready() {
  local response=""

  if command -v curl >/dev/null 2>&1; then
    curl --fail --silent --show-error --max-time 1 --noproxy '*' \
      "${API_URL}/api/v1/health" >/dev/null 2>&1
    return
  fi

  if exec 3<>/dev/tcp/127.0.0.1/8742 2>/dev/null; then
    printf 'GET /api/v1/health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n' >&3
    IFS= read -r response <&3 || true
    exec 3<&-
    exec 3>&-
    [[ "${response}" == *" 200 "* ]]
    return
  fi

  return 1
}

wait_for_api() {
  local api_pid="$1"
  local attempt

  for ((attempt = 0; attempt < 100; attempt++)); do
    if ! service_is_running "${api_pid}"; then
      printf 'error: the API stopped during startup.\n' >&2
      return 1
    fi
    if api_is_ready; then
      return 0
    fi
    sleep 0.1
  done

  printf 'error: the API did not become ready at %s.\n' "${API_URL}" >&2
  return 1
}

while (($# > 0)); do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    *)
      printf 'error: unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3))); then
  fail "Bash 4.3 or newer is required."
fi

cd "${PROJECT_ROOT}"
trap 'cleanup "$?"' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM
trap 'cleanup 129' HUP

require_command uv "uv is required; install it from https://docs.astral.sh/uv/."
require_command npm "npm is required; install the Node version listed in .nvmrc."
require_command setsid "setsid is required; install the util-linux package."

if ! uv run --frozen --no-sync epor version >/dev/null 2>&1; then
  fail "the EPOR environment is not ready; run the one-time setup commands in README.md."
fi
if [[ ! -x ui/node_modules/.bin/vite ]]; then
  fail "the UI dependencies are not ready; run 'cd ui && npm ci' once."
fi

printf 'Starting EPOR from %s\n' "${PROJECT_ROOT}"
# Each command performs its own covenant authorization before starting work.
start_service api uv run --frozen --no-sync epor api --host 127.0.0.1 --port 8742
api_pid="${SERVICE_PIDS[0]}"
if ! wait_for_api "${api_pid}"; then
  cleanup 1
fi

start_service worker uv run --frozen --no-sync epor worker
start_service ui uv run --frozen --no-sync epor ui

printf '\nConsole:  %s\n' "${UI_URL}"
printf 'API docs: %s/api/v1/docs\n' "${API_URL}"
printf 'Press Ctrl+C to stop all services.\n\n'

set +e
wait -n "${SERVICE_PIDS[@]}"
service_status=$?
set -e
if ((service_status == 0)); then
  service_status=1
fi
printf 'A service exited unexpectedly (status %s).\n' "${service_status}" >&2
cleanup "${service_status}"
