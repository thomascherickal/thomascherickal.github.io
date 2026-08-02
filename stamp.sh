#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
HASH=$(sha256sum style.css | cut -c1-12)
sed -i -E 's|(href="\./style\.css)(\?v=[^"]*)?"|\1?v='"$HASH"'"|g' *.html
echo "Stamped style.css?v=$HASH"
