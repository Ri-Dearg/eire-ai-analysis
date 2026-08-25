PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS outlet (
  id            INTEGER PRIMARY KEY,
  name          TEXT NOT NULL UNIQUE,
  category      TEXT NOT NULL CHECK (category IN ('legacy', 'counter-consensus')),
  base_url      TEXT NOT NULL UNIQUE,
  scrape_method TEXT
);

CREATE TABLE IF NOT EXISTS article (
  id             INTEGER PRIMARY KEY,
  outlet_id      INTEGER NOT NULL REFERENCES outlet(id),
  url            TEXT NOT NULL,
  author         TEXT,
  published_date TEXT,
  section        TEXT,
  body_text      TEXT,
  source_feed    TEXT,
  scraped_date   TEXT,
  url_canonical  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS raw_page (
  article_id   INTEGER PRIMARY KEY REFERENCES article(id),
  raw_html     TEXT NOT NULL,
  http_status  INTEGER,
  content_hash TEXT,
  fetched_date TEXT
);

INSERT OR IGNORE INTO outlet (id, name, category, base_url, scrape_method) VALUES
  (1, 'rte',            'legacy',            'https://www.rte.ie',           'sitemap'),
  (2, 'irish_examiner', 'legacy',            'https://www.irishexaminer.com','sitemap'),
  (3, 'the_liberal',    'counter-consensus', 'https://theliberal.ie',        'rss+sitemap'),
  (4, 'gript',          'counter-consensus', 'https://gript.ie',             'sitemap');
