"""Tools for scraping news outlets."""

from __future__ import annotations

from scraper.scraper import ingest

__all__ = ['ingest']


def __getattr__(name: str) -> object:
    """Resolve ingest on first access, not at package import time.

    Args:
        name (str): The attribute requested on the package.

    Returns:
        object: The :function scraper.scraper.ingest callable when requested.

    Raises:
        AttributeError: If name is not a lazily exposed attribute.

    """
    if name == 'ingest':
        return ingest
    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)
