"""Local dev server for the site with caching disabled (python -m http.server lets Chrome reuse stale modules).

    python scripts/serve.py [port]      -> http://127.0.0.1:8777/docs/
"""

from __future__ import annotations

import functools
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NoCache(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    handler = functools.partial(NoCache, directory=str(ROOT))
    print(f"serving {ROOT} at http://127.0.0.1:{port}/docs/ (no-cache)")
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
