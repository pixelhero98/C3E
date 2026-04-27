from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_gat_c3e_candidates import *  # noqa: F401,F403,E402
from tools.run_gat_c3e_candidates import main  # noqa: E402


if __name__ == "__main__":
    main()
