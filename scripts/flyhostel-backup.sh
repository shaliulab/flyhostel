#!/usr/bin/env bash
set -uo pipefail

REPO=/home/vibflysleep/opt/flyhostel/scripts
MAMBA_ENVIRONMENT="rapids-23.04"
PY=/home/vibflysleep/mambaforge/envs/${MAMBA_ENVIRONMENT}/bin/python
STATE_DIR="${STATE_DIRECTORY:-$HOME/.local/state/flyhostel-backup}"

JOBS=(
  "/media/vibflysleep/FLYHOSTEL1-2_HDD FlyHostel1"
  "/media/vibflysleep/FLYHOSTEL1-2_HDD FlyHostel2"
  "/media/vibflysleep/FLYHOSTEL3_HDD   FlyHostel3"
  "/media/vibflysleep/FLYHOSTEL4_HDD   FlyHostel4"
)

mkdir -p "$STATE_DIR"

# don't stack runs if a previous one is still copying
exec 9>"$STATE_DIR/.lock"
flock -n 9 || { echo "another run in progress, exiting"; exit 0; }

cd "$REPO" || exit 1

skipped=0
failed=0

for job in "${JOBS[@]}"; do
  read -r dest hostel <<<"$job"

  if ! mountpoint -q "$dest"; then
    echo "SKIP $hostel: $dest not mounted"
    skipped=$((skipped + 1))
    continue
  fi

  echo "=== $hostel -> $dest ==="
  if "$PY" main.py --dest "$dest" --flyhostel "$hostel"; then
    date -Is > "$STATE_DIR/$hostel.last"
  else
    echo "FAIL $hostel" >&2
    failed=$((failed + 1))
  fi
done

echo "summary: ${#JOBS[@]} jobs, $skipped skipped, $failed failed"

(( failed  > 0 )) && exit 1    # real error
(( skipped > 0 )) && exit 75   # incomplete: drives absent
exit 0