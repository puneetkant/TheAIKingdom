"""
Working Example 2: Web Scraping & APIs — Real Data from Public APIs
===================================================================
Demonstrates fetching live data from public APIs using urllib (stdlib only):
  1. REST API — Open-Meteo weather API (free, no key required)
  2. REST API — JSONPlaceholder (fake REST API for testing)
  3. HuggingFace Datasets API — download Titanic CSV
  4. HTML scraping — extract tables from Wikipedia with html.parser
  5. Rate limiting, retries, caching (cache to disk), error handling
  6. Paginates through an API endpoint

Run:  python working_example2.py
Note: Requires internet access (graceful fallback if offline).
"""
import html.parser
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


# -- Helper: safe HTTP GET with retry -----------------------------------------
def http_get(url: str, params: dict | None = None, retries: int = 3,
             timeout: int = 10) -> bytes | None:
    """Fetch URL with query params, retry on failure. Returns bytes or None."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ML-Learning/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.URLError as e:
            if attempt < retries:
                time.sleep(0.5 * attempt)
            else:
                print(f"  HTTP error: {e} -> offline fallback")
    return None


def http_get_json(url: str, params: dict | None = None) -> dict | list | None:
    raw = http_get(url, params)
    if raw is None: return None
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return None


# -- 1. Weather API (Open-Meteo) -----------------------------------------------
def demo_weather_api() -> None:
    print("=== Weather API (Open-Meteo) ===")
    data = http_get_json(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": 51.5,
            "longitude": -0.12,
            "daily": "temperature_2m_max,precipitation_sum",
            "forecast_days": 7,
            "timezone": "Europe/London",
        }
    )
    if data:
        dates  = data.get("daily", {}).get("time", [])
        temps  = data.get("daily", {}).get("temperature_2m_max", [])
        precip = data.get("daily", {}).get("precipitation_sum", [])
        print(f"  Location: lat={data.get('latitude')}, lon={data.get('longitude')}")
        print(f"  {'Date':<12} {'MaxTemp':>8} {'Precip':>8}")
        for d, t, p in zip(dates, temps, precip):
            print(f"  {d:<12} {t:>7.1f}°C {p:>7.1f}mm")
        # Save to cache
        cache = DATA / "weather_cache.json"
        cache.write_text(json.dumps(data, indent=2))
        print(f"  Cached to {cache.name}")
    else:
        print("  (offline — using synthetic data)")
        temps = [16.0, 17.5, 15.2, 18.0, 19.1, 14.8, 16.3]
        for i, t in enumerate(temps):
            print(f"  Day {i+1}: {t:.1f}°C")


# -- 2. JSONPlaceholder REST API (pagination) ----------------------------------
def demo_rest_api_pagination() -> None:
    print("\n=== REST API Pagination (JSONPlaceholder) ===")
    all_posts = []
    for page in range(1, 3):  # 2 pages
        data = http_get_json(
            "https://jsonplaceholder.typicode.com/posts",
            params={"_page": page, "_limit": 5}
        )
        if data is None: break
        all_posts.extend(data)
        print(f"  Page {page}: fetched {len(data)} posts")
        time.sleep(0.2)  # polite rate limiting

    if all_posts:
        print(f"  Total fetched: {len(all_posts)}")
        print("  First post:")
        p = all_posts[0]
        print(f"    id={p['id']}  userId={p['userId']}")
        print(f"    title: {p['title'][:60]}")
    else:
        print("  (offline — no posts fetched)")


# -- 3. HuggingFace dataset download (binary) ----------------------------------
def demo_hf_download() -> None:
    print("\n=== HuggingFace Dataset Download ===")
    dest = DATA / "titanic.csv"
    if dest.exists():
        print(f"  Cached: {dest.name} ({dest.stat().st_size:,} bytes)")
        return
    raw = http_get(
        "https://huggingface.co/datasets/phihung/titanic/resolve/main/train.csv"
    )
    if raw:
        dest.write_bytes(raw)
        print(f"  Downloaded: {dest.name} ({len(raw):,} bytes)")
    else:
        print("  (offline — creating synthetic CSV)")
        dest.write_text("PassengerId,Survived,Pclass,Name,Sex,Age\n"
                        "1,1,1,Alice,female,28\n2,0,3,Bob,male,35\n")


# -- 4. HTML table scraping (html.parser) --------------------------------------
class TableParser(html.parser.HTMLParser):
    """Minimal HTML parser that extracts text from <table> cells."""

    def __init__(self):
        super().__init__()
        self.in_cell = False
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []
        self._cell_text = ""

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self.in_cell = True
            self._cell_text = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.current_row.append(self._cell_text.strip())
            self.in_cell = False
        elif tag == "tr" and self.current_row:
            self.rows.append(self.current_row)
            self.current_row = []

    def handle_data(self, data):
        if self.in_cell:
            self._cell_text += data


def demo_html_scraping() -> None:
    print("\n=== HTML Scraping (html.parser) ===")
    # Use a stable, simple JSON-over-HTTP source instead of HTML scraping
    # (Wikipedia HTML changes frequently; use REST Countries API instead)
    data = http_get_json("https://restcountries.com/v3.1/region/europe?fields=name,population,area")
    if data and isinstance(data, list):
        countries = sorted(data, key=lambda c: c.get("population", 0), reverse=True)[:5]
        print(f"  Top 5 European countries by population:")
        print(f"  {'Country':<25} {'Population':>12} {'Area km²':>12}")
        print("  " + "-" * 52)
        for c in countries:
            name = c.get("name", {}).get("common", "?")
            pop  = c.get("population", 0)
            area = c.get("area", 0)
            print(f"  {name:<25} {pop:>12,} {area:>12,.0f}")
    else:
        print("  (offline — scraping example skipped)")
        # Show HTML parser demo with inline HTML
        sample_html = """
        <table>
          <tr><th>Country</th><th>Population</th></tr>
          <tr><td>Germany</td><td>83,000,000</td></tr>
          <tr><td>France</td><td>67,000,000</td></tr>
        </table>
        """
        parser = TableParser()
        parser.feed(sample_html)
        print("  Parsed inline HTML table:")
        for row in parser.rows:
            print(f"    {row}")


# -- 5. Caching and rate limiting demo ----------------------------------------
def demo_caching() -> None:
    print("\n=== Disk Cache Pattern ===")
    cache_file = DATA / "api_cache.json"

    def get_with_cache(url: str, cache_key: str) -> dict | None:
        cache = {}
        if cache_file.exists():
            cache = json.loads(cache_file.read_text())
        if cache_key in cache:
            print(f"  Cache HIT for {cache_key}")
            return cache[cache_key]

        data = http_get_json(url)
        if data:
            cache[cache_key] = data
            cache_file.write_text(json.dumps(cache))
            print(f"  Cache MISS — fetched and stored {cache_key}")
        return data

    get_with_cache("https://jsonplaceholder.typicode.com/todos/1", "todo_1")
    get_with_cache("https://jsonplaceholder.typicode.com/todos/1", "todo_1")  # should HIT


if __name__ == "__main__":
    demo_weather_api()
    demo_rest_api_pagination()
    demo_hf_download()
    demo_html_scraping()
    demo_caching()
