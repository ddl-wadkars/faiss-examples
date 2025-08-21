# make_embeddings_npy.py
# Generate emb_db.npy [N, d] and emb_q.npy [Q, d] (float32).
# Modes: synthetic (random) or text (Sentence-Transformers embeddings).

import argparse
import os
from pathlib import Path
import numpy as np

def save_npy(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr.astype(np.float32))
    print(f"Saved {path}  shape={arr.shape}  dtype={arr.dtype}")

# ---------------- Synthetic ----------------
def make_synthetic(N: int, Q: int, d: int, seed: int, normalize: bool):
    rng = np.random.RandomState(seed)
    xb = rng.randn(N, d).astype(np.float32)
    xq = rng.randn(Q, d).astype(np.float32)
    if normalize:
        # L2-normalize (useful for cosine similarity workflows)
        xb /= (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12)
        xq /= (np.linalg.norm(xq, axis=1, keepdims=True) + 1e-12)
    return xb, xq

# ---------------- Text ----------------
def load_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def embed_texts(texts, model_name: str, batch_size: int):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise SystemExit(
            "Missing dependency. Install with: pip install sentence-transformers"
        ) from e
    model = SentenceTransformer(model_name)
    # convert_to_numpy returns float32; we still cast to be explicit
    return np.asarray(
        model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True),
        dtype=np.float32,
    )

def make_text_embeddings(db_txt: Path, q_txt: Path, model_name: str, batch_size: int, normalize: bool):
    db_texts = load_lines(db_txt)
    q_texts  = load_lines(q_txt)
    if len(db_texts) == 0 or len(q_texts) == 0:
        raise SystemExit("Input text files must contain at least one non-empty line each.")
    xb = embed_texts(db_texts, model_name, batch_size)
    xq = embed_texts(q_texts,  model_name, batch_size)
    if normalize:
        xb /= (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12)
        xq /= (np.linalg.norm(xq, axis=1, keepdims=True) + 1e-12)
    return xb, xq

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Create emb_db.npy and emb_q.npy")
    sub = p.add_subparsers(dest="mode", required=True)

    # synthetic mode
    ps = sub.add_parser("synthetic", help="Generate random embeddings")
    ps.add_argument("--N", type=int, required=True, help="DB vectors")
    ps.add_argument("--Q", type=int, required=True, help="Query vectors")
    ps.add_argument("--d", type=int, required=True, help="Dimension")
    ps.add_argument("--seed", type=int, default=123)
    ps.add_argument("--normalize", action="store_true", help="L2-normalize rows")
    ps.add_argument("--out_dir", type=Path, default=Path("."))

    # text mode
    pt = sub.add_parser("text", help="Embed text files with Sentence-Transformers")
    pt.add_argument("--db_txt", type=Path, required=True, help="One document per line")
    pt.add_argument("--q_txt",  type=Path, required=True, help="One query per line")
    pt.add_argument("--model",  type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    pt.add_argument("--batch_size", type=int, default=512)
    pt.add_argument("--normalize", action="store_true", help="L2-normalize rows")
    pt.add_argument("--out_dir", type=Path, default=Path("."))

    return p.parse_args()

def main():
    args = parse_args()
    out_db = args.out_dir / "emb_db.npy"
    out_q  = args.out_dir / "emb_q.npy"

    if args.mode == "synthetic":
        xb, xq = make_synthetic(args.N, args.Q, args.d, args.seed, args.normalize)
    else:
        xb, xq = make_text_embeddings(args.db_txt, args.q_txt, args.model, args.batch_size, args.normalize)

    # Basic sanity checks
    if xb.ndim != 2 or xq.ndim != 2 or xb.shape[1] != xq.shape[1]:
        raise SystemExit(f"Bad shapes: xb={xb.shape}, xq={xq.shape} (dims must match)")
    if xb.dtype != np.float32 or xq.dtype != np.float32:
        xb = xb.astype(np.float32); xq = xq.astype(np.float32)

    save_npy(out_db, xb)
    save_npy(out_q,  xq)

    # Quick summary
    print(f"\nDone. emb_db.npy: {xb.shape}, emb_q.npy: {xq.shape}, d={xb.shape[1]}")
    print(f"Example norms — db mean={np.linalg.norm(xb,axis=1).mean():.3f}, q mean={np.linalg.norm(xq,axis=1).mean():.3f}")

if __name__ == "__main__":
    main()
