#!/usr/bin/env bash
# Fetch the two model source repos the annotate pass imports.
#
# Only source is cloned — weights come from the Hugging Face hub on first run, or
# from a local snapshot if you point the registry at one. Both repos are consumed as
# checkouts rather than installed packages because that is how upstream ships them.
#
#   scripts/setup_models.sh              # clone into ./third_party
#   CONTINUO_EXPRESSIVE_THIRD_PARTY=third_party scripts/setup_models.sh
#
# Idempotent: an existing checkout is left alone.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${CONTINUO_EXPRESSIVE_THIRD_PARTY:-$root/third_party}"
mkdir -p "$dest"

clone() {
  local url="$1" name="$2"
  if [ -d "$dest/$name/.git" ] || [ -d "$dest/$name/src" ]; then
    echo "== $name already present at $dest/$name"
  else
    echo "== cloning $name"
    git clone --depth 1 "$url" "$dest/$name"
  fi
}

clone https://github.com/tiantiaf0627/vox-profile-release.git vox-profile-release
clone https://github.com/tiantiaf0627/voxlect.git voxlect

cat <<EOF

Done. Both repos are in $dest.

If that is not ./third_party, export the paths the pipeline reads:
  export CONTINUO_EXPRESSIVE_VOXPROFILE_REPO=$dest/vox-profile-release
  export CONTINUO_EXPRESSIVE_VOXLECT_REPO=$dest/voxlect

Note: both expose their code as a top-level 'src' namespace package. They coexist on
sys.path only because neither ships src/__init__.py — see heads/loader.py.
EOF
