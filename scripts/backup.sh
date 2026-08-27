ROOT=/home/vibflysleep/flyhostel_data/videos
DEST_COMPUTER="cv3"

# flyhostel - group size
DATE_TIME="2026-08-25_14-00-00"
declare -A GROUP_SIZE_BY_FH=(
  [1]=3
  [2]=3
  [3]=3
  [4]=3
)

for i in 1 2 3 4;  do
  j="${GROUP_SIZE_BY_FH[$i]:-}"
  if [[ -z "$j" ]]; then
    echo "ERROR: No mapping for i=$i in GROUP_SIZE_BY_FH" >&2
    exit 1
  fi

  REL="FlyHostel${i}/${j}X/${DATE_TIME}"
  echo $REL

  cd "$ROOT" || exit 1

  # Parse REL datetime and compute cutoff = +1 day
  rel_stamp="${REL##*/}"                          # 2026-02-27_16-00-00
  rel_date="${rel_stamp%%_*}"                     # 2026-02-27
  rel_time="${rel_stamp#*_}"                      # 16-00-00
  rel_time="${rel_time//-/:}"                     # 16:00:00
  start="${rel_date} ${rel_time}"

  start_epoch=$(date -d "$start" +%s)                 || exit 1
  cutoff_epoch=$(( start_epoch + 27*3600 ))              # +1 hour
  cutoff=$(date -d "@$cutoff_epoch" '+%F %T')         || exit 1

  echo $start $cutoff

  find "$REL" -type f \
        -newermt "$start" \
      ! -newermt "$cutoff" \
        -mmin +10 \
      -print0 \
    | rsync -avzi \
        --from0 --files-from=- \
        --relative --no-implied-dirs \
        --keep-dirlinks \
        ./  ${DEST_COMPUTER:/flyhostel_data/videos/
done