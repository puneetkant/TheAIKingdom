# 1.2.8 Web Scraping & APIs

Fetch data from public APIs with stdlib `urllib`: weather, REST pagination, HuggingFace datasets, HTML parsing, disk caching.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Basic urllib, JSON parsing, simple GET request |
| `working_example2.py` | Open-Meteo weather API, JSONPlaceholder pagination, HuggingFace download, html.parser, disk cache |
| `working_example.ipynb` | Interactive: weather API → REST pagination → dataset download → caching |

## Run

```bash
python working_example.py
python working_example2.py   # requires internet (graceful offline fallback)
jupyter lab working_example.ipynb
```

## HTTP Quick Reference

```python
import urllib.request, urllib.parse, json

# GET with query params
params = {"q": "python", "format": "json"}
url = "https://api.example.com/search?" + urllib.parse.urlencode(params)
req = urllib.request.Request(url, headers={"User-Agent": "MyBot/1.0"})

with urllib.request.urlopen(req, timeout=10) as resp:
    data = json.loads(resp.read())

# POST with JSON body
body = json.dumps({"key": "value"}).encode()
req = urllib.request.Request(url, data=body,
      headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
```

## Rate Limiting + Retry Pattern

```python
import time, urllib.error

def http_get_json(url, retries=3, delay=0.5):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:          # Too Many Requests
                time.sleep(delay * 2**attempt)
            else:
                raise
    return None
```

## Free Public APIs (no key required)
| API | URL |
|-----|-----|
| Open-Meteo (weather) | `https://api.open-meteo.com/v1/forecast` |
| JSONPlaceholder (test) | `https://jsonplaceholder.typicode.com/` |
| REST Countries | `https://restcountries.com/v3.1/all` |
| HuggingFace Datasets | `https://huggingface.co/datasets/` |
| Open Library | `https://openlibrary.org/api/books` |

## Learning Resources
- [Python urllib docs](https://docs.python.org/3/library/urllib.html)
- [Real Python: Python Requests Library](https://realpython.com/python-requests/)
- [Requests library (3rd party)](https://requests.readthedocs.io/)
- [BeautifulSoup docs](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [MDN HTTP overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)
- **Book:** *Web Scraping with Python* (Ryan Mitchell)

- Try a small hands-on exercise focused on this topic.
- Keep the code in `project.py` in this folder.
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
