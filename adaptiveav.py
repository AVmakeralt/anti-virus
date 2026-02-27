#!/usr/bin/env python3
"""Top-level CLI launcher.  This script simply forwards execution to
`adaptiveav.page.main()` so that calls like `python adaptiveav.py scan ...`
continue to work after reorganising the package.
"""

from adaptiveav.page import main

if __name__ == "__main__":
    main()
