"""Sentence embeddings for the style-similarity measure and checks (proposal 4)."""
import numpy as np

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedder(device="cpu"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDER, device=device)


def embed_texts(model, texts, chunk_tokens=254):
    """Embed each text as the mean of its chunk embeddings, then L2-normalize.

    MiniLM reads at most 256 tokens, so a long reasoning trace would otherwise be
    represented by its opening only. We split every text into 254-token chunks
    (+2 special tokens = 256), embed each chunk, and average.
    Returns an array of shape (len(texts), dim).
    """
    tok = model.tokenizer
    chunks, owner = [], []
    for i, text in enumerate(texts):
        ids = tok(text or "", add_special_tokens=False)["input_ids"]
        for k in range(0, max(len(ids), 1), chunk_tokens):
            chunks.append(tok.decode(ids[k:k + chunk_tokens]))
            owner.append(i)
    E = model.encode(chunks, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    out = np.zeros((len(texts), E.shape[1]))
    np.add.at(out, owner, E)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
