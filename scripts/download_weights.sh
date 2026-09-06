#!/bin/bash
# =============================================================================
# Download the released pepDockQ model checkpoint from Zenodo into assets/.
#
# Usage:
#   scripts/download_weights.sh              # writes assets/model.pth
#   OUT=/tmp/model.pth scripts/download_weights.sh
#
# The script is idempotent: it skips the download if the target file already
# exists and its SHA-256 matches EXPECTED_SHA256.
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# Zenodo record (fill in once the record is public)
# -----------------------------------------------------------------------------
# TODO: replace with the real Zenodo direct-download URL, e.g.
#   https://zenodo.org/records/<record_id>/files/model.pth
ZENODO_URL="${ZENODO_URL:-https://zenodo.org/records/REPLACE_ME/files/model.pth}"

# TODO: replace with the SHA-256 of the released model.pth
#   Compute locally with:  sha256sum assets/model.pth
EXPECTED_SHA256="${EXPECTED_SHA256:-REPLACE_ME}"

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="${OUT:-$REPO_ROOT/assets/model.pth}"
mkdir -p "$(dirname "$OUT")"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "ERROR: neither sha256sum nor shasum is available" >&2
        exit 1
    fi
}

check_sha() {
    local file=$1
    local expected=$2
    if [[ "$expected" == "REPLACE_ME" ]]; then
        echo "[download_weights] WARNING: EXPECTED_SHA256 not set, skipping checksum verification"
        return 0
    fi
    local actual
    actual=$(sha256_of "$file")
    if [[ "$actual" != "$expected" ]]; then
        echo "ERROR: SHA-256 mismatch for $file" >&2
        echo "  expected: $expected" >&2
        echo "  got:      $actual"    >&2
        return 1
    fi
    echo "[download_weights] SHA-256 OK ($actual)"
}

# -----------------------------------------------------------------------------
# Skip if already downloaded and valid
# -----------------------------------------------------------------------------
if [[ -f "$OUT" ]]; then
    if check_sha "$OUT" "$EXPECTED_SHA256"; then
        echo "[download_weights] $OUT already present and valid — nothing to do."
        exit 0
    else
        echo "[download_weights] $OUT exists but checksum is wrong — re-downloading."
        rm -f "$OUT"
    fi
fi

if [[ "$ZENODO_URL" == *REPLACE_ME* ]]; then
    echo "ERROR: ZENODO_URL is not set." >&2
    echo "Edit scripts/download_weights.sh (or export ZENODO_URL) with the" >&2
    echo "Zenodo direct-download URL for model.pth."                        >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
echo "[download_weights] Fetching $ZENODO_URL -> $OUT"
if command -v curl >/dev/null 2>&1; then
    curl --fail --location --progress-bar --output "$OUT" "$ZENODO_URL"
elif command -v wget >/dev/null 2>&1; then
    wget --show-progress -O "$OUT" "$ZENODO_URL"
else
    echo "ERROR: need curl or wget to download weights" >&2
    exit 1
fi

check_sha "$OUT" "$EXPECTED_SHA256"
echo "[download_weights] Done. Checkpoint is at: $OUT"
