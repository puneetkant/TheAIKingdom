"""
Working Example: NLP Tools and Libraries
Covers NLTK, spaCy, HuggingFace Transformers, Gensim, and LangChain —
all as patterns + live demos where those libraries are installed.
"""
import re, os

# -- 1. NLTK -------------------------------------------------------------------
def nltk_overview():
    print("=== NLTK (Natural Language Toolkit) ===")
    print("  Install: pip install nltk")
    print()
    try:
        import nltk
        # download quietly if needed
        for resource in ["punkt", "stopwords", "averaged_perceptron_tagger",
                         "wordnet", "vader_lexicon"]:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                try: nltk.download(resource, quiet=True)
                except: pass
        print("  NLTK is installed. Running live demo:")
        text = "The quick brown fox jumps over the lazy dog. NLTK is a great library!"

        # Tokenisation
        try:
            tokens = nltk.word_tokenize(text)
        except Exception:
            tokens = text.split()
        print(f"  word_tokenize: {tokens[:8]}...")

        # POS tagging
        try:
            pos = nltk.pos_tag(tokens[:6])
            print(f"  pos_tag:       {pos}")
        except Exception:
            pass

        # Stopwords
        try:
            from nltk.corpus import stopwords
            sw = set(stopwords.words("english"))
            filtered = [t for t in tokens if t.lower() not in sw]
            print(f"  stopwords filtered: {filtered[:8]}")
        except Exception:
            pass

        # Sentiment (VADER)
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            score = sia.polarity_scores("This movie is absolutely fantastic!")
            print(f"  VADER sentiment: {score}")
        except Exception:
            pass

    except ImportError:
        print("  NLTK not installed — code patterns:")
        print("""
  import nltk
  nltk.download('punkt'); nltk.download('stopwords')

  tokens = nltk.word_tokenize("Hello, World!")
  pos    = nltk.pos_tag(tokens)
  chunks = nltk.ne_chunk(pos)

  stemmer    = nltk.stem.PorterStemmer()
  lemmatizer = nltk.stem.WordNetLemmatizer()

  from nltk.corpus import stopwords
  sw = stopwords.words('english')

  from nltk.sentiment import SentimentIntensityAnalyzer
  sia = SentimentIntensityAnalyzer()
  sia.polarity_scores("I love this!")  # compound, pos, neg, neu
        """)


# -- 2. spaCy ------------------------------------------------------------------
def spacy_overview():
    print("\n=== spaCy ===")
    print("  Install: pip install spacy && python -m spacy download en_core_web_sm")
    print()
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp("Apple was founded by Steve Jobs in Cupertino, California.")
            print("  spaCy is installed (en_core_web_sm). Running live demo:")
            print(f"  Tokens:    {[t.text for t in doc]}")
            print(f"  POS:       {[(t.text, t.pos_) for t in doc][:6]}")
            print(f"  Entities:  {[(e.text, e.label_) for e in doc.ents]}")
            print(f"  Sentences: {[s.text for s in doc.sents]}")
        except OSError:
            print("  spaCy installed but en_core_web_sm not downloaded.")
    except ImportError:
        pass

    print()
    print("  Code patterns:")
    print("""
  import spacy
  nlp = spacy.load('en_core_web_sm')
  doc = nlp("Apple was founded by Steve Jobs in Cupertino.")

  # Tokenisation, POS, lemma
  for token in doc:
      print(token.text, token.pos_, token.lemma_, token.is_stop)

  # Named entities
  for ent in doc.ents:
      print(ent.text, ent.label_)   # Apple ORG, Steve Jobs PERSON

  # Dependency parsing
  for token in doc:
      print(token.text, token.dep_, token.head.text)

  # Sentence segmentation
  for sent in doc.sents:
      print(sent.text)

  # Similarity (requires en_core_web_md or lg)
  doc1 = nlp("king"); doc2 = nlp("queen")
  print(doc1.similarity(doc2))
    """)

    print("  spaCy pipeline components:")
    components = ["tokenizer", "tagger (POS)", "parser (DEP)",
                  "ner", "lemmatizer", "textcat (optional)"]
    for c in components:
        print(f"    - {c}")


# -- 3. HuggingFace Transformers -----------------------------------------------
def huggingface_overview():
    print("\n=== HuggingFace Transformers ===")
    print("  Install: pip install transformers datasets tokenizers")
    print()
    try:
        from transformers import pipeline
        print("  Transformers is installed. Running sentiment pipeline:")
        clf = pipeline("sentiment-analysis",
                       model="distilbert-base-uncased-finetuned-sst-2-english",
                       truncation=True)
        texts = ["I love this!", "This is terrible."]
        for t, r in zip(texts, clf(texts)):
            print(f"    '{t}' -> {r['label']} ({r['score']:.4f})")
    except ImportError:
        pass

    print()
    print("  Pipeline tasks:")
    tasks = [
        ("sentiment-analysis",  "Text classification (pos/neg)"),
        ("text-generation",     "GPT-style autoregressive generation"),
        ("fill-mask",           "BERT-style masked token prediction"),
        ("ner",                 "Named entity recognition"),
        ("question-answering",  "Extractive QA (context + question)"),
        ("summarization",       "Abstractive summarisation (BART/T5)"),
        ("translation",         "Machine translation"),
        ("zero-shot-classification", "No-label text classification"),
        ("feature-extraction",  "Get embeddings"),
        ("image-classification","ViT, CLIP classification"),
    ]
    print(f"  {'Task':<28} Description")
    print(f"  {'-'*28} {'-'*35}")
    for t, d in tasks:
        print(f"  {t:<28} {d}")

    print()
    print("  Key APIs:")
    print("""
  from transformers import AutoTokenizer, AutoModelForSequenceClassification
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  model     = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

  # Datasets
  from datasets import load_dataset
  ds = load_dataset("imdb")
  ds["train"][0]   # {'text': '...', 'label': 1}

  # Trainer API
  from transformers import Trainer, TrainingArguments
  args = TrainingArguments(output_dir="./out", num_train_epochs=3, ...)
  trainer = Trainer(model=model, args=args, train_dataset=ds["train"])
  trainer.train()
    """)


# -- 4. Gensim ----------------------------------------------------------------
def gensim_overview():
    print("\n=== Gensim ===")
    print("  Install: pip install gensim")
    print("  Specialises in topic modelling and word vectors")
    print()
    try:
        import gensim
        from gensim.models import Word2Vec
        sentences = [
            ["the", "cat", "sat", "on", "the", "mat"],
            ["the", "dog", "barked", "at", "the", "cat"],
            ["a", "cat", "chased", "a", "dog"],
            ["machine", "learning", "models", "learn", "word", "vectors"],
        ]
        model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, epochs=50)
        print(f"  Gensim Word2Vec trained on {len(sentences)} sentences")
        if "cat" in model.wv and "dog" in model.wv:
            sim = model.wv.similarity("cat", "dog")
            print(f"  Similarity(cat, dog): {sim:.4f}")
    except ImportError:
        pass

    print("""
  Code patterns:
  from gensim.models import Word2Vec, FastText, KeyedVectors
  from gensim.models.ldamodel import LdaModel

  # Train Word2Vec
  model = Word2Vec(sentences, vector_size=100, window=5, min_count=5)
  model.wv['king']                          # get vector
  model.wv.most_similar('king', topn=5)     # nearest neighbours
  model.wv.similarity('cat', 'dog')         # cosine similarity

  # Load pre-trained (GloVe / fastText)
  wv = KeyedVectors.load_word2vec_format('glove.6B.100d.word2vec')

  # LDA topic model
  lda = LdaModel(corpus=bow_corpus, num_topics=10, id2word=dictionary)
  lda.print_topics()
    """)


# -- 5. Summary comparison ----------------------------------------------------
def library_comparison():
    print("\n=== NLP Library Comparison ===")
    rows = [
        ("NLTK",           "Research/education",   "All classic NLP algos",     "Slow, academic API"),
        ("spaCy",          "Production NLP",        "Fast, industrial-grade",    "Less customisable"),
        ("HuggingFace",    "BERT/GPT/T5 tasks",     "Largest model hub",         "Heavier dependencies"),
        ("Gensim",         "Topic models, W2V",     "Efficient embeddings",      "Narrower scope"),
        ("Flair",          "Sequence labelling",    "State-of-art NER/POS",      "Slower than spaCy"),
        ("stanza (NLP)",   "Multilingual",          "CoNLL-award winning",       "Java-like API"),
        ("LangChain",      "LLM applications",      "Chain LLMs + tools",        "Fast-moving API"),
    ]
    print(f"  {'Library':<16} {'Best for':<22} {'Strength':<28} Weakness")
    print(f"  {'-'*16} {'-'*22} {'-'*28} {'-'*25}")
    for r in rows:
        print(f"  {r[0]:<16} {r[1]:<22} {r[2]:<28} {r[3]}")


if __name__ == "__main__":
    nltk_overview()
    spacy_overview()
    huggingface_overview()
    gensim_overview()
    library_comparison()
