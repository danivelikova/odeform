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
  [checkpoints]="1x-9U7LHkom5MA0VUkpKVH6PlC1TGUmiL"
  [bottle]="1MFDyrhsZWyXyEztrmdw2JY6iMtWfruUb"
  [cat]="1YFBxUfS09OHNexHxqxOEsiA10sGuwJlc"
  [dog]="1ZmRSJ-4iL_GjOP99vP29Noomcln_RotZ"
  [donut]="1K_kwQ-OhujRFrqEUvC1-hRNiq83i7PUz"
  [doritos]="1Vwx-Ho2enADI_c-KK8lwFw_3GRaoRu06"
  [flipflop]="1Yl93TfdQe9gtRmiRTvqhJTgLSSXe54j0"
  [pillow]="16TJvCZyym-xm22DqUtrcvlzrbSueLJKT"
  [mass_elastic]="1-NGkyylTp7EsPY9wAANzs0UOdzU8o8aL"
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
