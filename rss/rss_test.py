"""Test feasibility of retrieving content via RSS feeds."""

from typing import Any

import requests

# Define Agent for testing RSS feeds.
USER_AGENT: str = (
    'CapstoneResearchBot/0.1 '
    '(HDip Data Analytics, DBS; contact: 20074605@mydbs.ie) '
    'Mozilla/5.0 compatible'
)
HEADERS: dict[str, str] = {'User-Agent': USER_AGENT}
TIMEOUT: int = 20

# News outlets to test. Note the mix of mainstream and fringe sources.
FEEDS: dict[str, str] = {
    'The Journal': 'https://www.thejournal.ie/feed/',
    'Irish Examiner': 'https://feeds.feedburner.com/ietopstories',
    'Gript': 'https://gript.ie/feed/',
    'The Liberal': 'https://theliberal.ie/feed/',
}


def check_rss_feed(name: str, url: str):
    print(f'Test: {name}: {url}')

    result: dict[str, Any] = {'name': name, 'url': url, 'ok': False}

    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        print(f'HTTP status: {response.status_code}')
        result['http_status'] = response.status_code
        response.raise_for_status()
    except Exception as e:
        print(f'FETCH FAILED: {e}')
        result['error'] = str(e)
    print(result)
