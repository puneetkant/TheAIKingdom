"""
Working Example: Web Scraping & APIs
Covers HTTP requests, REST API consumption, JSON parsing,
HTML scraping with BeautifulSoup, rate limiting, and pagination.
Uses only stdlib + requests + beautifulsoup4 (common installs).
"""
import json
import time
import urllib.request
import urllib.parse
from datetime import datetime


# -- 1. urllib (stdlib, no extra deps) ----------------------------------------
def urllib_demo():
    print("=== 1. urllib (stdlib) ===")
    url = "https://httpbin.org/get?source=urllib&demo=true"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            print(f"  status   : {resp.status}")
            print(f"  url      : {data['url']}")
            print(f"  args     : {data['args']}")
            print(f"  user-agent: {data['headers'].get('User-Agent','')[:50]}")
    except Exception as e:
        print(f"  (network unavailable): {e}")


# -- 2. requests library -------------------------------------------------------
def requests_demo():
    print("\n=== 2. requests ===")
    try:
        import requests

        # GET
        r = requests.get("https://httpbin.org/get",
                         params={"q": "python", "page": 1},
                         timeout=8)
        r.raise_for_status()
        data = r.json()
        print(f"  GET status  : {r.status_code}")
        print(f"  GET url     : {data['url']}")

        # POST JSON
        payload = {"username": "alice", "score": 99}
        r = requests.post("https://httpbin.org/post",
                          json=payload, timeout=8)
        data = r.json()
        print(f"\n  POST status : {r.status_code}")
        print(f"  POST json   : {data['json']}")

        # Headers
        headers = {"Authorization": "Bearer demo-token", "X-App-ID": "42"}
        r = requests.get("https://httpbin.org/headers", headers=headers, timeout=8)
        print(f"\n  custom headers sent: {r.json()['headers'].get('X-App-Id')}")

        # Session (reuse connection, persist cookies)
        with requests.Session() as session:
            session.headers.update({"User-Agent": "MyBot/1.0"})
            r = session.get("https://httpbin.org/cookies/set?key=demo&value=123",
                            timeout=8, allow_redirects=True)
            print(f"\n  session cookie: {session.cookies.get('key')}")

    except ImportError:
        print("  requests not installed — pip install requests")
    except Exception as e:
        print(f"  (network unavailable): {e}")


# -- 3. Public REST API — JSONPlaceholder --------------------------------------
def rest_api_demo():
    print("\n=== 3. REST API (JSONPlaceholder) ===")
    base = "https://jsonplaceholder.typicode.com"
    try:
        import requests

        # List posts
        r = requests.get(f"{base}/posts", params={"_limit": 3}, timeout=8)
        posts = r.json()
        for p in posts:
            title = p["title"][:45]
            print(f"  post {p['id']}: {title}...")

        # Get single resource
        r = requests.get(f"{base}/users/1", timeout=8)
        user = r.json()
        print(f"\n  user: {user['name']} <{user['email']}>")
        print(f"    company: {user['company']['name']}")
        print(f"    lat,lng: {user['address']['geo']['lat']},{user['address']['geo']['lng']}")

        # Create (POST)
        new_post = {"title": "My AI Post", "body": "Deep learning rocks", "userId": 1}
        r = requests.post(f"{base}/posts", json=new_post, timeout=8)
        print(f"\n  created post id: {r.json()['id']}  status: {r.status_code}")

        # Update (PUT)
        updated = new_post | {"id": 1}
        r = requests.put(f"{base}/posts/1", json=updated, timeout=8)
        print(f"  updated status  : {r.status_code}")

        # Delete
        r = requests.delete(f"{base}/posts/1", timeout=8)
        print(f"  deleted status  : {r.status_code}")

    except ImportError:
        print("  requests not installed — pip install requests")
    except Exception as e:
        print(f"  (network unavailable): {e}")


# -- 4. BeautifulSoup — HTML scraping -----------------------------------------
def beautifulsoup_demo():
    print("\n=== 4. BeautifulSoup HTML Parsing ===")
    html = """
    <html>
    <head><title>Python Resources</title></head>
    <body>
      <h1>Learning Python</h1>
      <ul id="links">
        <li><a href="https://python.org">Python Official</a></li>
        <li><a href="https://docs.python.org">Python Docs</a></li>
        <li><a href="https://pypi.org">PyPI</a></li>
      </ul>
      <table class="scores">
        <tr><th>Library</th><th>Stars</th></tr>
        <tr><td>NumPy</td><td>25k</td></tr>
        <tr><td>Pandas</td><td>42k</td></tr>
        <tr><td>scikit-learn</td><td>58k</td></tr>
      </table>
    </body>
    </html>
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        print(f"  title   : {soup.title.string}")
        print(f"  h1      : {soup.h1.string}")

        # Extract links
        print("\n  links:")
        for a in soup.select("#links a"):
            print(f"    {a['href']}  ->  {a.string}")

        # Extract table
        print("\n  table:")
        for row in soup.select("table.scores tr"):
            cells = [td.get_text(strip=True) for td in row.find_all(["th","td"])]
            print(f"    {cells}")

        # Find by text
        pypi = soup.find("a", string="PyPI")
        print(f"\n  PyPI link: {pypi['href']}")

    except ImportError:
        print("  beautifulsoup4 not installed — pip install beautifulsoup4")


# -- 5. Rate limiting & pagination --------------------------------------------
def rate_limiting_demo():
    print("\n=== 5. Rate Limiting & Pagination ===")
    try:
        import requests

        def paginated_fetch(base_url, total_pages=3, delay=0.3):
            all_items = []
            for page in range(1, total_pages + 1):
                r = requests.get(base_url,
                                 params={"_page": page, "_limit": 5},
                                 timeout=8)
                r.raise_for_status()
                items = r.json()
                if not items:
                    break
                all_items.extend(items)
                print(f"  page {page}: fetched {len(items)} items")
                time.sleep(delay)   # respect rate limit
            return all_items

        items = paginated_fetch("https://jsonplaceholder.typicode.com/posts")
        print(f"  total fetched: {len(items)} posts")
        print(f"  titles: {[i['title'][:30] for i in items[:3]]}")

    except ImportError:
        print("  requests not installed — pip install requests")
    except Exception as e:
        print(f"  (network unavailable): {e}")


# -- 6. Error handling for network code ---------------------------------------
def robust_request():
    print("\n=== 6. Robust Request Pattern ===")
    def fetch_with_retry(url, max_retries=3, backoff=1.0):
        try:
            import requests
            for attempt in range(1, max_retries + 1):
                try:
                    r = requests.get(url, timeout=8)
                    r.raise_for_status()
                    return r.json()
                except requests.exceptions.Timeout:
                    print(f"  attempt {attempt}: timeout")
                except requests.exceptions.HTTPError as e:
                    print(f"  attempt {attempt}: HTTP {e.response.status_code}")
                    if e.response.status_code < 500:
                        raise   # don't retry 4xx
                except requests.exceptions.ConnectionError:
                    print(f"  attempt {attempt}: connection error")
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
            return None
        except ImportError:
            print("  requests not installed")
            return None

    data = fetch_with_retry("https://jsonplaceholder.typicode.com/todos/1")
    if data:
        print(f"  fetched: {data}")
    else:
        print("  all retries exhausted")


if __name__ == "__main__":
    urllib_demo()
    requests_demo()
    rest_api_demo()
    beautifulsoup_demo()
    rate_limiting_demo()
    robust_request()
