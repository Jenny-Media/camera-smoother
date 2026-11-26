import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoother.__main__ import main

raise SystemExit(main())
