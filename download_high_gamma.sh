#!/bin/bash
set -u
# Downloads the High Gamma Dataset's train/test .mat files directly from
# GIN's raw file server (the server that actually holds the git-annexed
# content -- confirmed via `curl -I`, no git-annex/datalad needed; see
# data_high_gamma.py's module docstring). Cloning
# https://web.gin.g-node.org/robintibor/high-gamma-dataset with plain
# `git clone` instead only gets you git-annex pointer stub files, NOT
# the real ~500MB-per-file data -- that is not what this script does.
#
# Unlike a bare `wget -q` loop, this validates each downloaded file's
# HDF5 magic-byte signature and retries on failure/mismatch instead of
# silently leaving a corrupt/HTML-error-page file in place with the
# right filename -- that failure mode surfaces later as a cryptic h5py
# "file signature not found" error at training time instead of at
# download time.
#
# Usage:
#   OUT_DIR=/kaggle/working/hgd/data SUBJECTS="1 2" ./download_high_gamma.sh
#   OUT_DIR=/kaggle/working/hgd/data ./download_high_gamma.sh   # all 14
: "${OUT_DIR:?Set OUT_DIR, e.g. /kaggle/working/hgd/data}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14}"
BASE_URL="https://gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data"

mkdir -p "$OUT_DIR/train" "$OUT_DIR/test"

is_valid_hdf5() {
    [ -s "$1" ] && [ "$(head -c 8 "$1" | od -An -tx1 | tr -d ' \n')" = "894844460d0a1a0a" ]
}

download_one() {
    local url="$1" out="$2"
    for attempt in 1 2 3; do
        curl -L -f -s -S -A "Mozilla/5.0" "$url" -o "$out"
        if is_valid_hdf5 "$out"; then
            echo "  OK: $out ($(stat -c%s "$out" 2>/dev/null || stat -f%z "$out") bytes)"
            return 0
        fi
        echo "  attempt $attempt: invalid/failed download for $out (size=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out" 2>/dev/null || echo 0)), retrying..."
        sleep 2
    done
    echo "FAILED to download a valid HDF5 file: $url -> $out" >&2
    return 1
}

status=0
for i in $SUBJECTS; do
    echo "=== subject $i ==="
    download_one "$BASE_URL/train/${i}.mat" "$OUT_DIR/train/${i}.mat" || status=1
    download_one "$BASE_URL/test/${i}.mat"  "$OUT_DIR/test/${i}.mat"  || status=1
done

if [ "$status" -eq 0 ]; then
    echo "All requested subjects downloaded and HDF5-signature-validated under $OUT_DIR"
else
    echo "One or more files FAILED validation after 3 attempts -- see above." >&2
fi
exit $status
