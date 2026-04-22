"""Test feasibility of retrieving content via RSS feeds.
"""

# Define Agent for testing RSS feeds.
USER_AGENT: str = (
    "CapstoneResearchBot/0.1 "
    "(HDip Data Analytics, DBS; contact: 20074605@mydbs.ie) "
    "Mozilla/5.0 compatible"
)
HEADERS: dict[str, str] = {"User-Agent": USER_AGENT}
TIMEOUT: int = 20

# News outlets to test. Note the mix of mainstream and fringe sources.
FEEDS: dict[str, str] = {
    "The Journal":    "https://www.thejournal.ie/feed/",
    "Irish Examiner": "https://feeds.feedburner.com/ietopstories",
    "Gript":          "https://gript.ie/feed/",
    "The Liberal":    "https://theliberal.ie/feed/",
}