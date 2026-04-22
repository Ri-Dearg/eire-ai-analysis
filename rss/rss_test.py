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


def check_rss_feed(name: str, url: str) -> dict[str, Any]:
    """Check an RSS feed and parse the returned information.

    Args:
        name (str): Name of the nws outlet.
        url (str): URL of RSS feed.

    Returns:
        dict[str, Any]: Dictionary of information about response.

    """
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

    # All dates, to see how far back the feed reaches - Suggested by Chatgpt
    dates = [e.get('published', '') for e in entries if e.get('published')]
    if dates:
        print(f'date range: {dates[-1]} -> {dates[0]}')

    # Update result with details so they can be printed later.
    result.update(
        {
            'has_summary': has_summary,
            'has_content': has_content,
            'summary_len': summary_len,
            'content_len': content_len,
            'sample_link': sample.get('link'),
            'sample_title': sample.get('title'),
            'oldest_in_feed': dates[-1] if dates else None,
            'newest_in_feed': dates[0] if dates else None,
        }
    )
    return result


def check_article(url: str) -> dict[str, Any]:
    print(f'Article: {url}')
    result: dict[str, Any] = {'url': url, 'ok': False}

    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        print(f'HTTP status: {response.status_code}')
        result['http_status'] = response.status_code
        response.raise_for_status()
    except Exception as e:
        print(f'FETCH FAILED: {e}')
        result['error'] = str(e)
    print(result)
    return result
