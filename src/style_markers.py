"""Each model's distinctive phrases, computed the same way for every model.

Used to fill the "Common Language" slot of the paper's forgery prompt (make_stimuli.py).
Method: weighted log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi
& Quinn 2008) on 2-3-word n-grams of the rule-decision traces, model vs. the others.
The same number of phrases (K) is taken for every model, so no style gets a bigger dose.

Filters (identical for all models):
  * the n-gram occurs >= 15 times overall and >= 10 times for this model
  * it contains at least one word
  * no hedge words (the content checks forbid hedging, so suggesting them would only
    produce rejected rewrites)
  * no near-duplicates (an n-gram contained in an already chosen one, or vice versa)
"""
import collections
import math
import re

HEDGE_WORDS = {"maybe", "if", "whether", "perhaps", "probably", "might", "seems", "unsure",
               "not", "wondering", "suppose", "assume", "believe"}


def _tokens(text):
    return re.findall(r"[a-z']+|[.,!?:;]", text.lower())


def _ngrams(tokens, ns=(2, 3)):
    for n in ns:
        for i in range(len(tokens) - n + 1):
            yield tuple(tokens[i:i + n])


def _display(gram):
    """('.', 'so', ',') -> 'So,' ; ('we', 'can') -> 'we can'."""
    words = list(gram)
    start = False
    while words and words[0] in ".,!?:;":
        start = words.pop(0) in ".!?"
        start = True if start else start
    text = " ".join(words)
    text = re.sub(r" ([.,!?:;])", r"\1", text)
    return text[:1].upper() + text[1:] if start and text else text


def _is_style_gram(g):
    """2-3 real words, or one word + comma (a discourse marker like 'So,' / 'However,').
    Rejects bare sentence-initial words ('. we') that carry no usable phrasing."""
    words = [w for w in g if re.match(r"[a-z']", w)]
    if len(words) >= 2:
        return True
    return len(words) == 1 and g[-1] == "," and g[-2] == words[0]


def distinctive_phrases(traces_by_model, k=12, min_total=15, min_model=10, min_traces=8):
    """traces_by_model: {model: [reasoning text, ...]} -> {model: [phrase, ...]}

    min_traces: the phrase must appear in at least this many of the model's traces.
    Style markers recur across prompts; topic phrases ("no flash", "the grammar")
    pile up in one or two long traces.
    """
    counts = {m: collections.Counter(g for t in texts for g in _ngrams(_tokens(t)))
              for m, texts in traces_by_model.items()}
    doc_freq = {m: collections.Counter(g for t in texts for g in set(_ngrams(_tokens(t))))
                for m, texts in traces_by_model.items()}
    prior = collections.Counter()
    for c in counts.values():
        prior.update(c)
    a0 = sum(prior.values()) * 0.01
    out = {}
    for m in counts:
        rest = collections.Counter()
        for o in counts:
            if o != m:
                rest.update(counts[o])
        n1, n2 = sum(counts[m].values()), sum(rest.values())
        scored = []
        for g, total in prior.items():
            y1, y2 = counts[m][g], rest[g]
            if total < min_total or y1 < min_model or doc_freq[m][g] < min_traces:
                continue
            if not _is_style_gram(g) or HEDGE_WORDS & set(g):
                continue
            a = total * 0.01
            delta = (math.log((y1 + a) / (n1 + a0 - y1 - a))
                     - math.log((y2 + a) / (n2 + a0 - y2 - a)))
            z = delta / math.sqrt(1 / (y1 + a) + 1 / (y2 + a))
            scored.append((z, g))
        scored.sort(reverse=True)
        chosen = []
        for z, g in scored:
            words = " ".join(w for w in g if re.match(r"[a-z']", w))
            if any(words in c or c in words for c, _ in chosen):
                continue
            chosen.append((words, g))
            if len(chosen) == k:
                break
        out[m] = [_display(g) for _, g in chosen]
    return out
