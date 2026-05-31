"""``python -m loris`` entry point.

Delegates to the chase pipeline orchestrator. The full flag set is defined in
:func:`loris.pipeline.orchestrator.parse_args`; run ``python -m loris --help``
to list options.
"""

from __future__ import annotations

from loris.pipeline.orchestrator import main

if __name__ == "__main__":
    main()
