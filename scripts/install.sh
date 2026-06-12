#!/usr/bin/env bash
#
# Build garde (release) and put it on PATH.
#
# SwiftPM executables locate their resource bundles (XGBoost model JSON, MLX
# Metal shaders) relative to the real binary, so instead of copying the binary
# out of the build tree this installs a tiny wrapper at ~/.local/bin/garde that
# execs the build product in place. Rebuilds are picked up automatically; rerun
# this script only if you move the repo.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/.build/release/garde"
DEST_DIR="$HOME/.local/bin"
WRAPPER="$DEST_DIR/garde"

echo "Building garde (release)…"
swift build -c release --package-path "$REPO"

mkdir -p "$DEST_DIR"
cat > "$WRAPPER" <<EOF
#!/bin/sh
exec "$BIN" "\$@"
EOF
chmod +x "$WRAPPER"
echo "Installed wrapper: $WRAPPER → $BIN"

case ":$PATH:" in
    *":$DEST_DIR:"*)
        echo "~/.local/bin already on PATH — done. Try: garde --help"
        ;;
    *)
        RC="$HOME/.zshrc"
        LINE='export PATH="$HOME/.local/bin:$PATH"'
        if [ -f "$RC" ] && grep -qxF "$LINE" "$RC"; then
            echo "PATH line already in $RC — restart your shell, then try: garde --help"
        else
            printf '\n%s\n' "$LINE" >> "$RC"
            echo "Added ~/.local/bin to PATH in $RC — restart your shell, then try: garde --help"
        fi
        ;;
esac
