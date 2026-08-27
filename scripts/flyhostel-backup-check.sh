#!/usr/bin/env bash
set -uo pipefail

STATE_DIR="${STATE_DIRECTORY:-$HOME/.local/state/flyhostel-backup}"
MAX_AGE_DAYS=3
HOSTELS=(FlyHostel1 FlyHostel2 FlyHostel3 FlyHostel4)

stale=()
for h in "${HOSTELS[@]}"; do
  f="$STATE_DIR/$h.last"
  if [[ ! -f "$f" ]] || [[ -n "$(find "$f" -mtime +"$MAX_AGE_DAYS" -print -quit)" ]]; then
    stale+=("$h")
  fi
done

if (( ${#stale[@]} > 0 )); then
  msg="No successful backup in ${MAX_AGE_DAYS}d: ${stale[*]}"
  echo "$msg" >&2
  notify-send -u critical "FlyHostel backup" "$msg" 2>/dev/null
  exit 1
fi

echo "all hostels backed up within ${MAX_AGE_DAYS}d"