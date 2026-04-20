"""
Working Example 2: NLP Tools and Libraries — NLTK, spaCy, sklearn, HuggingFace overview
=========================================================================================
Demonstrates capabilities of major NLP libraries.

Run:  python working_example2.py
"""
from pathlib import Path
import re, string

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

TEXT = "The quick brown fox jumped over the lazy dogs. Natural language processing is fascinating!"

def demo_nltk():
    try:
        import nltk
        try: nltk.data.find("tokenizers/punkt")
        except: nltk.download("punkt", quiet=True)
        tokens = nltk.word_tokenize(TEXT)
        print(f"  NLTK tokenize ({len(tokens)} tokens): {tokens[:8]}")
    except ImportError:
        # Fallback: simple regex tokenizer
        tokens = re.findall(r"\b\w+\b", TEXT)
        print(f"  Regex tokenize ({len(tokens)} tokens): {tokens[:8]}")

def demo_sklearn():
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = [TEXT, "Machine learning and NLP are closely related.", "Deep learning models process text."]
    vect = TfidfVectorizer()
    X = vect.fit_transform(corpus)
    print(f"  sklearn TF-IDF: {X.shape[0]} docs × {X.shape[1]} features")

def demo_regex():
    """Regex patterns common in NLP."""
    email_pat = r"[\w.+-]+@[\w-]+\.[a-zA-Z]+"
    url_pat   = r"https?://[^\s]+"
    test = "Contact user@example.com or visit https://nlp.org for details."
    emails = re.findall(email_pat, test); urls = re.findall(url_pat, test)
    print(f"  Emails: {emails}  URLs: {urls}")

def demo_tool_matrix():
    rows = [
        ("NLTK", "Tokenization, stemming, tagging, corpora", "Teaching, research"),
        ("spaCy", "Fast NER, dep-parse, pipelines", "Production NLP"),
        ("sklearn", "TF-IDF, classifiers, pipelines", "ML benchmarks"),
        ("HuggingFace", "BERT, GPT, fine-tuning", "State-of-the-art"),
        ("Gensim", "Word2Vec, topic models", "Embeddings, LDA"),
    ]
    print("\n  NLP Library Comparison:")
    print(f"  {'Library':15s} {'Key Features':40s} {'Best Use':25s}")
    print("  " + "-"*80)
    for r in rows:
        print(f"  {r[0]:15s} {r[1]:40s} {r[2]:25s}")

def demo_text_similarity():
    """Compute pairwise sentence similarity with TF-IDF cosine."""
    print("\n=== Text Similarity (TF-IDF Cosine) ===")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sentences = [
        "Deep learning revolutionises natural language processing.",
        "Neural networks transform NLP research and applications.",
        "The stock market closed higher on Friday.",
        "Machine learning is a subset of artificial intelligence.",
    ]
    vect = TfidfVectorizer()
    mat = vect.fit_transform(sentences)
    sim = cosine_similarity(mat)
    header = f"  {'':35s}" + "  ".join(f"S{i+1}" for i in range(4))
    print(header)
    for i, row in enumerate(sim):
        print(f"  S{i+1} {sentences[i][:30]:30s}  " + "  ".join(f"{v:.2f}" for v in row))


def demo_preprocessing_pipeline():
    """Full pipeline: clean -> tokenize -> remove stopwords -> vectorise."""
    print("\n=== Full Preprocessing Pipeline ===")
    from sklearn.feature_extraction.text import CountVectorizer
    corpus = [TEXT, "Natural language processing uses machine learning.",
              "Text classification identifies categories in documents."]
    cleaned = [re.sub(r"[^\w\s]", "", t).lower() for t in corpus]
    cv = CountVectorizer(ngram_range=(1, 2), max_features=20)
    X = cv.fit_transform(cleaned)
    print(f"  Count vectorised: {X.shape[0]} docs x {X.shape[1]} (1-2)-gram features")
    print(f"  Top features: {cv.get_feature_names_out()[:10].tolist()}")


if __name__ == "__main__":
    print("=== NLP Tools and Libraries ===\n")
    demo_nltk()
    demo_sklearn()
    demo_regex()
    demo_tool_matrix()
    demo_text_similarity()
    demo_preprocessing_pipeline()
