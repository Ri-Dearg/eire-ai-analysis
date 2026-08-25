"""Launch the dashboard: ``python -m dashboard``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from streamlit.web import cli

logger = logging.getLogger(__name__)

APP = Path(__file__).resolve().parent / 'app.py'


def main(argv: list[str] | None = None) -> int:
    """Start the Streamlit server on the dashboard entry point.

    Args:
        argv (list[str] | None): Extra ``streamlit run`` arguments, e.g.
            ``['--server.port', '8502']``. Defaults to ``sys.argv[1:]``.

    Returns:
        int: The exit code from Streamlit.

    """
    extra = sys.argv[1:] if argv is None else argv
    logger.info('starting streamlit on %s', APP)
    sys.argv = ['streamlit', 'run', str(APP), *extra]
    return int(cli.main(standalone_mode=False) or 0)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    raise SystemExit(main())
