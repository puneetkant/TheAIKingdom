"""
Working Example: Classical NLP Models
Covers Naive Bayes for text classification, Hidden Markov Models (HMM)
for POS tagging, CRF concepts, and n-gram language models.
"""
import math, re
from collections import Counter, defaultdict
import numpy as np


# -- helpers -------------------------------------------------------------------
def tokenise(text):
    return re.sub(r"[^\w\s]", "", text.lower()).split()


# -- 1. Naive Bayes Text Classifier -------------------------------------------
class NaiveBayesTextClassifier:
    """Multinomial Naive Bayes for text classification."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha   # Laplace smoothing

    def fit(self, X_texts, y):
        self.classes_  = sorted(set(y))
        self.log_prior = {}
        self.log_lhood = {}
        self.vocab_    = Counter()

        n = len(y)
        for text in X_texts:
            for w in tokenise(text):
                self.vocab_[w] += 1

        vocab_size = len(self.vocab_)
        class_docs  = defaultdict(list)
        for text, label in zip(X_texts, y):
            class_docs[label].append(text)

        for c in self.classes_:
            self.log_prior[c] = math.log(len(class_docs[c]) / n)
            word_cnt = Counter()
            for text in class_docs[c]:
                word_cnt.update(tokenise(text))
            total = sum(word_cnt.values()) + self.alpha * vocab_size
            self.log_lhood[c] = {
                w: math.log((word_cnt[w] + self.alpha) / total)
                for w in self.vocab_
            }
            # OOV log-likelihood
            self.log_lhood[c]["<UNK>"] = math.log(self.alpha / total)

    def predict_proba(self, text):
        tokens = tokenise(text)
        scores = {}
        for c in self.classes_:
            s = self.log_prior[c]
            for t in tokens:
                s += self.log_lhood[c].get(t, self.log_lhood[c]["<UNK>"])
            scores[c] = s
        # Convert to probability (log-sum-exp)
        max_s = max(scores.values())
        exp_s = {c: math.exp(s - max_s) for c, s in scores.items()}
        total = sum(exp_s.values())
        return {c: v / total for c, v in exp_s.items()}

    def predict(self, text):
        return max(self.predict_proba(text), key=self.predict_proba(text).get)


def naive_bayes_demo():
    print("=== Naive Bayes Text Classification ===")
    train = [
        ("I love this movie, it was great!", "positive"),
        ("Fantastic film, highly recommended!", "positive"),
        ("Amazing story and beautiful cinematography", "positive"),
        ("The acting was superb and the plot was engaging", "positive"),
        ("Wonderful experience, one of the best movies!", "positive"),
        ("Terrible movie, complete waste of time", "negative"),
        ("Awful story with poor acting", "negative"),
        ("Boring and predictable, I hated it", "negative"),
        ("The worst film I have seen this year", "negative"),
        ("Disgusting and offensive content throughout", "negative"),
    ]
    X_tr, y_tr = zip(*train)

    clf = NaiveBayesTextClassifier(alpha=1.0)
    clf.fit(list(X_tr), list(y_tr))

    tests = [
        "This is a wonderful movie",
        "I hated every minute of this terrible film",
        "Average story, neither good nor bad",
    ]
    print(f"  {'Text':<42} {'Predicted':<10} P(pos) P(neg)")
    print(f"  {'-'*42} {'-'*10} {'-'*6} {'-'*6}")
    for text in tests:
        probs = clf.predict_proba(text)
        pred  = max(probs, key=probs.get)
        print(f"  {text[:42]:<42} {pred:<10} {probs['positive']:.3f}  {probs['negative']:.3f}")

    train_acc = sum(clf.predict(x) == y for x, y in zip(X_tr, y_tr)) / len(y_tr)
    print(f"\n  Training accuracy: {train_acc:.2f}")
    print(f"  Key properties: fast, no iterations, good baseline for text")


# -- 2. N-gram Language Model ---------------------------------------------------
class NgramLM:
    """Smoothed n-gram language model with Laplace smoothing."""
    def __init__(self, n=2, alpha=0.01):
        self.n     = n
        self.alpha = alpha
        self.counts = defaultdict(Counter)  # context -> next_word -> count
        self.vocab  = set()

    def fit(self, tokens):
        tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        self.vocab = set(tokens)
        for i in range(len(tokens) - self.n + 1):
            ctx  = tuple(tokens[i:i+self.n-1])
            next_w = tokens[i+self.n-1]
            self.counts[ctx][next_w] += 1
        return self

    def logprob(self, context, word):
        ctx   = tuple(context[-(self.n-1):]) if self.n > 1 else ()
        cnt   = self.counts[ctx]
        num   = cnt[word] + self.alpha
        denom = sum(cnt.values()) + self.alpha * len(self.vocab)
        return math.log(num / denom)

    def perplexity(self, tokens):
        tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        log_prob = 0
        N = 0
        for i in range(self.n - 1, len(tokens)):
            ctx  = tokens[i-self.n+1:i] if self.n > 1 else []
            word = tokens[i]
            log_prob += self.logprob(ctx, word)
            N += 1
        return math.exp(-log_prob / max(N, 1))

    def generate(self, seed, max_len=20, rng=None):
        rng = rng or np.random.default_rng(0)
        tokens = list(seed)
        for _ in range(max_len):
            ctx = tuple(tokens[-(self.n-1):]) if self.n > 1 else ()
            if ctx not in self.counts: break
            words = list(self.counts[ctx].keys())
            freqs = np.array([self.counts[ctx][w] + self.alpha for w in words])
            probs = freqs / freqs.sum()
            next_w = rng.choice(words, p=probs)
            if next_w == "</s>": break
            tokens.append(next_w)
        return tokens


def ngram_lm_demo():
    print("\n=== N-gram Language Model ===")
    corpus = """
    the cat sat on the mat the cat sat on the hat
    the dog sat on the mat the dog barked at the cat
    the man saw the cat and the dog the woman fed the cat
    a cat is not a dog a dog is not a cat words have meaning
    language models predict the next word in a sequence
    """.split()

    rng = np.random.default_rng(5)
    print(f"  Corpus: {len(corpus)} tokens")
    print()
    for n in [1, 2, 3]:
        lm = NgramLM(n=n, alpha=0.01).fit(corpus)
        pp = lm.perplexity(corpus[:30])
        generated = lm.generate(["the"], max_len=10, rng=rng)
        print(f"  {n}-gram LM: perplexity={pp:.2f}")
        print(f"    Generated: {' '.join(generated)}")


# -- 3. Hidden Markov Model for POS tagging -----------------------------------
class HMM:
    """First-order HMM for sequence labelling (POS tagging)."""
    def __init__(self):
        self.tags     = []
        self.trans    = defaultdict(Counter)   # P(tag | prev_tag)
        self.emit     = defaultdict(Counter)   # P(word | tag)
        self.init     = Counter()              # P(first tag)
        self.alpha    = 0.001

    def fit(self, sentences):
        """sentences: list of list of (word, tag)"""
        for sent in sentences:
            tags_seq = [tag for _, tag in sent]
            words    = [w   for w,  _  in sent]
            self.init[tags_seq[0]] += 1
            for i, (w, t) in enumerate(sent):
                self.emit[t][w.lower()] += 1
                if i > 0:
                    self.trans[tags_seq[i-1]][t] += 1
        self.tags = list(set(t for s in sentences for _, t in s))

    def viterbi(self, words):
        """Viterbi decoding."""
        n = len(words)
        T = len(self.tags)
        t2i = {t: i for i, t in enumerate(self.tags)}

        # dp[t][i] = log P(best path ending at tag t, word i)
        dp   = np.full((T, n), -1e18)
        back = np.full((T, n), -1, dtype=int)

        total_init = sum(self.init.values()) + self.alpha * T
        for ti, tag in enumerate(self.tags):
            emit_p  = (self.emit[tag][words[0].lower()] + self.alpha) / \
                      (sum(self.emit[tag].values()) + self.alpha * 1000)
            init_p  = (self.init[tag] + self.alpha) / total_init
            dp[ti, 0] = math.log(init_p) + math.log(emit_p)

        for i in range(1, n):
            for ti, tag in enumerate(self.tags):
                emit_p = (self.emit[tag][words[i].lower()] + self.alpha) / \
                         (sum(self.emit[tag].values()) + self.alpha * 1000)
                best_prev, best_s = -1, -1e18
                for pi, prev_tag in enumerate(self.tags):
                    total_trans = sum(self.trans[prev_tag].values()) + self.alpha * T
                    trans_p = (self.trans[prev_tag][tag] + self.alpha) / total_trans
                    s = dp[pi, i-1] + math.log(trans_p) + math.log(emit_p)
                    if s > best_s:
                        best_s, best_prev = s, pi
                dp[ti, i] = best_s
                back[ti, i] = best_prev

        # Trace back
        path = [int(dp[:, -1].argmax())]
        for i in range(n-1, 0, -1):
            path.insert(0, back[path[0], i])
        return [self.tags[ti] for ti in path]


def hmm_demo():
    print("\n=== HMM POS Tagging ===")
    train_sentences = [
        [("the", "DT"), ("cat", "NN"), ("sat", "VBD"), ("on", "IN"), ("the", "DT"), ("mat", "NN")],
        [("a", "DT"), ("dog", "NN"), ("barked", "VBD"), ("loudly", "RB")],
        [("she", "PRP"), ("runs", "VBZ"), ("fast", "RB")],
        [("the", "DT"), ("quick", "JJ"), ("brown", "JJ"), ("fox", "NN"), ("jumped", "VBD")],
        [("he", "PRP"), ("saw", "VBD"), ("a", "DT"), ("beautiful", "JJ"), ("woman", "NN")],
        [("dogs", "NNS"), ("and", "CC"), ("cats", "NNS"), ("are", "VBP"), ("friends", "NNS")],
        [("the", "DT"), ("man", "NN"), ("runs", "VBZ"), ("very", "RB"), ("fast", "RB")],
        [("she", "PRP"), ("saw", "VBD"), ("the", "DT"), ("dog", "NN")],
    ]
    hmm = HMM()
    hmm.fit(train_sentences)

    tests = [
        ["the", "dog", "barked"],
        ["she", "runs", "fast"],
        ["a", "quick", "fox"],
    ]
    print(f"  {'Words':<30} Predicted POS")
    print(f"  {'-'*30} {'-'*25}")
    for words in tests:
        preds = hmm.viterbi(words)
        print(f"  {str(words):<30} {preds}")

    print()
    print("  Tags: DT=Determiner  NN=Noun  VBD=Past Verb  IN=Preposition")
    print("        RB=Adverb  JJ=Adjective  PRP=Pronoun  NNS=Noun Plural")


# -- 4. CRF concepts -----------------------------------------------------------
def crf_concepts():
    print("\n=== Conditional Random Fields (CRF) ===")
    print("  HMM: generative P(X, Y) — models joint distribution")
    print("  CRF: discriminative P(Y|X) — models conditional directly")
    print()
    print("  Linear-chain CRF score:")
    print("    P(Y|X) ∝ exp(Sigma_t Sigma_k lambda_k · f_k(y_{t-1}, y_t, X, t))")
    print()
    print("  Features f_k can depend on the whole input X:")
    print("    - Current word, previous word, next word")
    print("    - Word shape (ALL_CAPS, Title, digit)")
    print("    - Prefix/suffix of the word")
    print("    - Previous label, current label combination")
    print()
    print("  Training: maximise log-likelihood via gradient descent (L-BFGS)")
    print("  Inference: Viterbi algorithm (same as HMM)")
    print()
    print("  CRF beats HMM because:")
    print("    1. Can use arbitrary overlapping features")
    print("    2. No independence assumption on observations")
    print("    3. Discriminative -> higher accuracy")
    print()
    print("  Packages: sklearn-crfsuite, python-crfsuite, spaCy (uses CRF internally)")


if __name__ == "__main__":
    naive_bayes_demo()
    ngram_lm_demo()
    hmm_demo()
    crf_concepts()
