#!/usr/bin/env bash
#
# Rebuild garde (release) and refresh the ~/.local/bin wrapper + PATH entry.
#
# End state is identical to scripts/install.sh (which is idempotent); this
# exists as the explicit "update my install" entry point and adds:
#
#   --clean   wipe .build first for a from-scratch build — useful when SwiftPM
#             caches go stale (e.g. after moving the repo or editing the local
#             Frigate dependency's package structure)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"

if [ "${1:-}" = "--clean" ]; then
    echo "Removing $REPO/.build for a clean rebuild…"
    rm -rf "$REPO/.build"
fi

exec "$REPO/scripts/install.sh"
