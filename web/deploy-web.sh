#!/bin/bash
set -e
cd "$(dirname "$0")"
npm run build

pkill -f "serve -s dist -l 4003" 2>/dev/null || true
sleep 1

nohup npx serve -s dist -l 4003 > /tmp/furkanca-web-serve.log 2>&1 &
sleep 2

ACTUAL_PID=$(pgrep -f "node .*serve -s dist -l 4003" | head -1)
if [ -z "$ACTUAL_PID" ]; then
  echo "❌ serve process did not start — check /tmp/furkanca-web-serve.log"
  exit 1
fi
echo "$ACTUAL_PID" > /tmp/furkanca-web-serve.pid

echo "✅ Deployed: serve process running (PID $ACTUAL_PID) on 127.0.0.1:4003, nginx proxies app.furkanca.algorixstudio.com to it."
echo "   Verify: curl -sS -o /dev/null -w 'HTTP %{http_code}\n' https://app.furkanca.algorixstudio.com/"
