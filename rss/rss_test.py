"""Test feasibility of retrieving content via RSS feeds."""

import json
import datetime
from pathlib import Path
from typing import Any

import feedparser
import requests
import trafilatura
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


def _fetch(url: str, result: dict[str, Any]) -> requests.Response | None:
    """Fetch a url for parsing.

    Args:
        url (str): Url to fetch.
        result (dict[str, Any]): Dictionary with info about URL.

    Returns:
        requests.Response | None: Either return response or nothing on error.

    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        print(f'HTTP status: {response.status_code}')
        result['http_status'] = response.status_code
        response.raise_for_status()
    except RequestException as e:
        print(f'FETCH FAILED: {e}')
        result['error'] = str(e)
        return None
    return response


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

    if (response := _fetch(url, result)) is None:
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
    """Check to se if article content is readable.

    Args:
        url (str): URL for the article.

    Returns:
        dict[str, Any]: Results dictionary with info.

    """
    print(f'Article: {url}')
    result: dict[str, Any] = {'url': url, 'ok': False}

    if (response := _fetch(url, result)) is None:
        return result

    text = trafilatura.extract(response.text, include_comments=False) or ''
    word_count = len(text.split())
    result['ok'] = True
    result['char_count'] = len(text)
    result['word_count'] = word_count

    if not text:
        print('Extractor returned nothing, issue?')
        return result

    print(f'  Extracted {len(text):,} chars, {word_count:,} words')
    print(f'  First 240 chars: {text[:240].strip()!r}')
    return result


def main() -> None:

    feed_results = [check_rss_feed(name, url) for name, url in FEEDS.items()]

    print('------- Article body extraction')
    article_results = [
        check_article(result['sample_link'])
        for result in feed_results
        if result.get('ok') and result.get('sample_link')
    ]

    print('------- SUMMARY')
    for result in feed_results:
        if not result.get('ok'):
            print(f'{result["name"]}: {result.get("error", "unknown error")}')
            continue
        body_in_rss = 'yes' if result.get('content_len', 0) > 1000 else 'no'
        print(
            f'✅ {result["name"]}: {result["entries"]:>3} items  |  '
            f'body in RSS? {body_in_rss}  |  '
            f'oldest item: {result.get("oldest_in_feed", "?")}'
        )

    # Persist for later inspection / inclusion in the proposal appendix
    out_path = Path('rss_feasibility_results.json')
    out_path.write_text(
        json.dumps(
            {
                'run_at': datetime.datetime.now(tz=datetime.UTC).isoformat(
                    timespec='seconds'
                ),
                'feeds': feed_results,
                'articles': article_results,
            },
            indent=2,
            default=str,
        ),
        encoding='utf-8',
    )
    print('Operation complete!')


if __name__ == '__main__':
    main()
