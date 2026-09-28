#!/usr/bin/env bash
# Download pretrained checkpoints and datasets from Google Drive and unpack them into
# ./checkpoints and ./data.
#
# Usage:
#   bash scripts/download_assets.sh                   # checkpoints only (39 MB)
#   bash scripts/download_assets.sh checkpoints donut mass_elastic
#   bash scripts/download_assets.sh all               # checkpoints + all datasets
#
# Available assets: checkpoints, bottle, cat, dog, donut, doritos, flipflop, pillow,
#                   contact_force (= all seven objects), mass_elastic, all
set -euo pipefail

FOLDER_URL="https://drive.google.com/drive/folders/1FmD-MZsBkA3cIzFlTwfNlf-u1JqdCRyy"

declare -A FILE_ID=(
  [checkpoints]="__ID_odeform_checkpoints__"
  [bottle]="__ID_contact_force_bottle__"
  [cat]="__ID_contact_force_cat__"
  [dog]="__ID_contact_force_dog__"
  [donut]="__ID_contact_force_donut__"
  [doritos]="__ID_contact_force_doritos__"
  [flipflop]="__ID_contact_force_flipflop__"
  [pillow]="__ID_contact_force_pillow__"
  [mass_elastic]="__ID_mass_elastic__"
)
declare -A FILE_NAME=(
  [checkpoints]="odeform_checkpoints.zip"
  [mass_elastic]="mass_elastic.zip"
)
OBJECTS=(bottle cat dog donut doritos flipflop pillow)
for o in "${OBJECTS[@]}"; do FILE_NAME[$o]="contact_force_$o.zip"; done

command -v gdown >/dev/null || { echo "gdown not found: pip install gdown"; exit 1; }
cd "$(dirname "$0")/.."
mkdir -p downloads

requested=("$@")
[ ${#requested[@]} -eq 0 ] && requested=(checkpoints)
assets=()
for a in "${requested[@]}"; do
  case "$a" in
    all) assets+=(checkpoints "${OBJECTS[@]}" mass_elastic) ;;
    contact_force) assets+=("${OBJECTS[@]}") ;;
    *) [ -n "${FILE_ID[$a]:-}" ] || { echo "Unknown asset: $a"; exit 1; }; assets+=("$a") ;;
  esac
done

for a in "${assets[@]}"; do
  zip="downloads/${FILE_NAME[$a]}"
  if [ ! -f "$zip" ]; then
    echo ">> Downloading $a -> $zip"
    gdown "https://drive.google.com/uc?id=${FILE_ID[$a]}" -O "$zip" || {
      echo "Download failed. You can also download ${FILE_NAME[$a]} manually from"
      echo "  $FOLDER_URL"
      echo "and place it in ./downloads/"; exit 1; }
  fi
  echo ">> Extracting $zip"
  unzip -q -o "$zip" -d .
done
echo "Done."
