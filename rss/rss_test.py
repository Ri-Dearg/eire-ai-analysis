"""Test feasibility of retrieving content via RSS feeds."""

from typing import Any

import feedparser
import requests
from requests.exceptions import RequestException

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

    # Create dictionary to report information
    result: dict[str, Any] = {'name': name, 'url': url, 'ok': False}

    # Make a call to url and parse response
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        print(f'HTTP status: {response.status_code}')
        result['http_status'] = response.status_code
        response.raise_for_status()
    # Deal with errors
    except RequestException as e:
        print(f'FETCH FAILED: {e}')
        result['error'] = str(e)
        return result

    # Parse the response and entries
    feed = feedparser.parse(response.content)
    entries = feed.entries
    print(f'Items returned: {len(entries)}')
    result['entries'] = len(entries)
    result['ok'] = True

    if not entries:
        return result

    # Take the first item in entries and see what is in it.
    sample = entries[0]
    has_content = 'content' in sample
    has_summary = 'summary' in sample
    summary_len = len(sample.get('summary', ''))  # pyright: ignore[reportArgumentType]
    content_len = len(sample['content'][0].get('value', '')) if has_content else 0  # pyright: ignore[reportArgumentType]

    # Check some of the more useful fields
    print(f'title:     {sample.get("title", "<missing>")}')
    print(f'link:      {sample.get("link", "<missing>")}')
    print(f'published: {sample.get("published", "<missing>")}')
    print(f'summary length: {summary_len:,} chars')
    print(f'content length: {content_len:,} chars')
    print(result)
    return result
