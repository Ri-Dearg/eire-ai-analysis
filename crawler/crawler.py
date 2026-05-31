"""WIP."""

import logging

logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
USER_AGENT = (
    'CapstoneResearchBot/0.1 (HDip Data Analytics, DBS; contact: 20074605@mydbs.ie)'
)
HEADERS = {'User-Agent': USER_AGENT}
# Request timeout in seconds.
REQUEST_TIMEOUT = 20

# Small delay between sitemap fetches to the same outlet.
INTER_REQUEST_DELAY = 0.5

# Standard sitemap locations to try, in order (same as the probe).
SITEMAP_CANDIDATES = (
    '/sitemap_index.xml',  # Yoast SEO (most common on WordPress)
    '/sitemap.xml',  # Plain default
    '/sitemaps/sitemap.xml',  # Some sites put them in a subfolder
    '/sitemap-index/44-google_sitemap.xml',
    '/wp-sitemap.xml',  # WordPress core (5.5+)
    '/news-sitemap.xml',  # Google News sitemap convention
)
