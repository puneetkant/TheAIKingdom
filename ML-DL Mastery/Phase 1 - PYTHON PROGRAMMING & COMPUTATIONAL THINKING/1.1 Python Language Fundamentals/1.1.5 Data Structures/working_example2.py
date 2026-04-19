"""
Working Example 2: Data Structures — Real-World Data Science Patterns
======================================================================
Uses the MovieLens 1M dataset structure (simulated / downloaded) to
demonstrate practical use of lists, dicts, sets, and collections for
recommendation-system style data processing.

Run:  python working_example2.py
"""
import urllib.request
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter, deque, namedtuple
from heapq import nlargest, nsmallest

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# -- 1. Download a small movie ratings CSV from HuggingFace -------------------
RATINGS_URL = (
    "https://huggingface.co/datasets/Shahrukh0/MovieLens-Small/resolve/main/ratings.csv"
)
MOVIES_URL  = (
    "https://huggingface.co/datasets/Shahrukh0/MovieLens-Small/resolve/main/movies.csv"
)


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    print(f"Downloading {dest.name} …")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [OK] {dest.stat().st_size // 1024} KB")
    except Exception as e:
        print(f"  [X] {e}. Using synthetic data.")
        if "ratings" in dest.name:
            dest.write_text(
                "userId,movieId,rating,timestamp\n"
                + "\n".join(
                    f"{u},{m},{(u*m)%5+1:.1f},946684800"
                    for u in range(1, 30)
                    for m in range(1, 20)
                )
            )
        else:
            dest.write_text(
                "movieId,title,genres\n"
                "1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy\n"
                "2,Jumanji (1995),Adventure|Children|Fantasy\n"
                "3,Grumpier Old Men (1995),Comedy|Romance\n"
            )
    return dest


# -- 2. Lists — sort, slice, comprehension -------------------------------------
def demo_lists(ratings: list[dict]) -> None:
    print("=== Lists ===")
    scores = [float(r["rating"]) for r in ratings[:1000]]

    # Sort
    top10     = sorted(scores, reverse=True)[:10]
    bottom10  = sorted(scores)[:10]
    print(f"  Top-10 ratings    : {top10}")
    print(f"  Bottom-10 ratings : {bottom10}")

    # Slicing
    print(f"  Middle 5 (500-505): {scores[500:505]}")

    # List as stack
    stack = []
    for s in scores[:5]:
        stack.append(s)
    print(f"  Stack push 5 items: {stack}")
    print(f"  Stack pop         : {stack.pop()} -> remaining {stack}")

    # List as queue (deque is faster for queue)
    queue = deque(maxlen=5)
    for s in scores[:8]:
        queue.append(s)
    print(f"  Deque (maxlen=5) after 8 pushes: {list(queue)}")


# -- 3. Dicts — inverted index, merging ----------------------------------------
def demo_dicts(ratings: list[dict], movies: list[dict]) -> dict:
    print("\n=== Dicts ===")
    # Build movie lookup: {movieId: title}
    movie_map = {row["movieId"]: row["title"] for row in movies}

    # Aggregate ratings per movie
    movie_ratings: dict[str, list[float]] = defaultdict(list)
    for r in ratings:
        movie_ratings[r["movieId"]].append(float(r["rating"]))

    # Compute mean rating per movie
    movie_stats = {
        mid: {"title": movie_map.get(mid, f"Movie {mid}"),
              "mean":  sum(rs) / len(rs),
              "count": len(rs)}
        for mid, rs in movie_ratings.items()
        if len(rs) >= 5   # at least 5 ratings
    }

    # Top 5 by mean rating
    top5 = nlargest(5, movie_stats.items(), key=lambda x: x[1]["mean"])
    print("  Top 5 movies by mean rating:")
    for mid, s in top5:
        print(f"    {s['title'][:35]:<35} mean={s['mean']:.2f}  n={s['count']}")

    # Dict merge (Python 3.9+ union operator)
    extra = {"new_key": "new_value"}
    merged = movie_stats | {"_meta": {"total": len(movie_stats)}} | extra
    print(f"  Dict merge: total keys = {len(merged)}")

    return movie_stats


# -- 4. Sets — genre analysis --------------------------------------------------
def demo_sets(movies: list[dict]) -> None:
    print("\n=== Sets ===")
    all_genres: set[str] = set()
    movie_genres: dict[str, set] = {}

    for m in movies:
        genres = set(m.get("genres", "").split("|"))
        genres.discard("(no genres listed)")
        movie_genres[m["movieId"]] = genres
        all_genres |= genres   # set union

    print(f"  Total unique genres: {len(all_genres)}")

    # Genres in common between two movies
    ids = list(movie_genres.keys())[:2]
    if len(ids) >= 2:
        a, b = movie_genres[ids[0]], movie_genres[ids[1]]
        print(f"  Movie {ids[0]} genres : {a}")
        print(f"  Movie {ids[1]} genres : {b}")
        print(f"  Intersection (n)   : {a & b}")
        print(f"  Union (u)          : {a | b}")
        print(f"  Difference (a-b)   : {a - b}")


# -- 5. Counter + namedtuple ---------------------------------------------------
def demo_counter_namedtuple(ratings: list[dict]) -> None:
    print("\n=== Counter & namedtuple ===")
    Rating = namedtuple("Rating", ["user_id", "movie_id", "score"])
    typed  = [Rating(r["userId"], r["movieId"], float(r["rating"])) for r in ratings[:500]]

    # Counter
    score_dist = Counter(t.score for t in typed)
    print("  Rating distribution:")
    for score in sorted(score_dist):
        bar = "#" * (score_dist[score] // 5)
        print(f"    {score}: {bar} ({score_dist[score]})")

    # Most active users
    user_counts = Counter(t.user_id for t in typed)
    print(f"  Most active users (top 5): {user_counts.most_common(5)}")


# -- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    r_path = download(RATINGS_URL, DATA_DIR / "ratings.csv")
    m_path = download(MOVIES_URL,  DATA_DIR / "movies.csv")

    with open(r_path, newline="", encoding="utf-8") as f:
        ratings = list(csv.DictReader(f))
    with open(m_path, newline="", encoding="utf-8") as f:
        movies = list(csv.DictReader(f))

    print(f"Loaded {len(ratings):,} ratings, {len(movies):,} movies\n")
    demo_lists(ratings)
    demo_dicts(ratings, movies)
    demo_sets(movies)
    demo_counter_namedtuple(ratings)
