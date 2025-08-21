# faiss_step1_cosine_or_synthetic.py
# Toggle between synthetic Gaussian data and real text embeddings.
# Adds cosine/IP support, vector stats, and hubness check.
import os, time, math, tempfile, argparse
from typing import Tuple, Dict, Any, List
import numpy as np

try:
    import faiss
except ImportError as e:
    raise SystemExit("Install FAISS: `pip install faiss-cpu` (or faiss-gpu).") from e

# ----------------------------
# Defaults (overridable via CLI)
# ----------------------------
DEFAULTS = dict(
    seed=123,
    mode=None,                 # "synthetic" or "real" (auto-infer if None)
    metric="l2",               # "cosine" or "l2"
    d=768,                     # synthetic only
    nb=200_000,                # synthetic db size
    nq=2_000,                  # synthetic query size
    k=10,
    ivf_nlist=4096,
    train_size=100_000,
    nprobe_sweep=[1, 8, 16, 32],
    batch_size=256,
    # Real embeddings inputs (choose ONE source: npy OR text)
    real_db_npy=None,          # path to emb_db.npy [N,d]
    real_q_npy=None,           # path to emb_q.npy  [Q,d]
    texts_db_path=None,        # txt file, one doc per line
    texts_q_path=None,         # txt file, one query per line
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
)

# ----------------------------
# Data loaders
# ----------------------------
def make_synthetic(nb, nq, d, seed):
    rng = np.random.RandomState(seed)
    xb = rng.randn(nb, d).astype(np.float32)
    xq = rng.randn(nq, d).astype(np.float32)
    return xb, xq

def load_npy(db_path, q_path):
    xb = np.load(db_path).astype(np.float32)
    xq = np.load(q_path).astype(np.float32)
    return xb, xq

def load_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def embed_texts(texts, model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise SystemExit(
            "`sentence-transformers` needed for on-the-fly embedding:\n"
            "  pip install sentence-transformers"
        ) from e
    # GPU if available; otherwise CPU
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts, convert_to_numpy=True, show_progress_bar=True
    ).astype(np.float32)
    return emb

def get_data(cfg):
    if cfg["mode"] == "synthetic":
        xb, xq = make_synthetic(cfg["nb"], cfg["nq"], cfg["d"], cfg["seed"])
    else:
        if cfg["real_db_npy"] and cfg["real_q_npy"]:
            if not (os.path.exists(cfg["real_db_npy"]) and os.path.exists(cfg["real_q_npy"])):
                raise SystemExit("Provided .npy paths do not exist.")
            xb, xq = load_npy(cfg["real_db_npy"], cfg["real_q_npy"])
        elif cfg["texts_db_path"] and cfg["texts_q_path"]:
            if not (os.path.exists(cfg["texts_db_path"]) and os.path.exists(cfg["texts_q_path"])):
                raise SystemExit("Provided text paths do not exist.")
            db_texts = load_texts(cfg["texts_db_path"])
            q_texts  = load_texts(cfg["texts_q_path"])
            if not db_texts or not q_texts:
                raise SystemExit("Text files must contain at least one non-empty line each.")
            xb = embed_texts(db_texts, cfg["embed_model"])
            xq = embed_texts(q_texts,  cfg["embed_model"])
        else:
            raise SystemExit("For --mode real, pass either --real-db-npy/--real-q-npy or --texts-db/--texts-q.")
    return xb.astype(np.float32), xq.astype(np.float32)

# ----------------------------
# Cosine/IP helpers
# ----------------------------
def maybe_normalize_for_cosine(xb, xq, metric):
    if metric == "cosine":
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)
    return xb, xq

def build_flat(xb, metric):
    d = xb.shape[1]
    index = faiss.IndexFlatIP(d) if metric == "cosine" else faiss.IndexFlatL2(d)
    index.add(xb)
    return index

def build_ivf_flat(xb, metric, nlist, train_size, seed):
    d = xb.shape[1]
    quant = faiss.IndexFlatIP(d) if metric == "cosine" else faiss.IndexFlatL2(d)
    m = faiss.METRIC_INNER_PRODUCT if metric == "cosine" else faiss.METRIC_L2
    ivf = faiss.IndexIVFFlat(quant, d, nlist, m)
    rs = np.random.RandomState(seed)
    train_idx = rs.choice(xb.shape[0], size=min(train_size, xb.shape[0]), replace=False)
    ivf.train(xb[train_idx])
    ivf.add(xb)
    return ivf

# ----------------------------
# Metrics & search
# ----------------------------
def ground_truth_flat(xb, xq, k, metric):
    idx = build_flat(xb, metric)
    D, I = idx.search(xq, k)
    return D, I

def search_batched(index, xq, k, batch):
    n = xq.shape[0]
    perq_ms = []
    allI, allD = [], []
    for s in range(0, n, batch):
        e = min(s + batch, n)
        t0 = time.perf_counter()
        D, I = index.search(xq[s:e], k)
        dt = (time.perf_counter() - t0) * 1000.0 / (e - s)
        allD.append(D); allI.append(I)
        perq_ms.extend([dt] * (e - s))
    D = np.vstack(allD); I = np.vstack(allI)
    return {"D": D, "I": I, "p50_ms": float(np.median(perq_ms)), "p95_ms": float(np.percentile(perq_ms, 95))}

def recall_at_k(I_pred, I_true, k):
    nn_true = I_true[:, 0]
    hit = np.any(I_pred[:, :k] == nn_true[:, None], axis=1).mean()
    overlap = [len(set(I_pred[i, :k]) & set(I_true[i, :k])) / k for i in range(I_true.shape[0])]
    return float(hit), float(np.mean(overlap))

# ----------------------------
# Diagnostic stats for real embeddings
# ----------------------------
def print_embedding_stats(xb, xq):
    norms_b = np.linalg.norm(xb, axis=1)
    norms_q = np.linalg.norm(xq, axis=1)
    print(f"DB norms: mean={norms_b.mean():.3f}  std={norms_b.std():.3f}  min={norms_b.min():.3f}  max={norms_b.max():.3f}")
    print(f"Q  norms: mean={norms_q.mean():.3f}  std={norms_q.std():.3f}")

    # cosine spread on a sample
    samp = min(2000, xb.shape[0])
    idx = np.random.default_rng(0).choice(xb.shape[0], size=samp, replace=False)
    Xn = xb[idx].copy()
    faiss.normalize_L2(Xn)
    sims = (Xn @ Xn.T).astype(np.float32)
    up = sims[np.triu_indices_from(sims, 1)]
    print(f"Pairwise cosine (sample {samp}): mean={up.mean():.3f} std={up.std():.3f}  p95={np.percentile(up,95):.3f}")

    # hubness: how often a DB vector appears in others' top-k (tiny run)
    d_small = min(5000, xb.shape[0])
    q_small = min(1000, xq.shape[0])
    k_hub = max(1, min(10, d_small))  # don’t ask for more neighbors than exist
    idx_small = build_flat(xb[:d_small], "cosine")
    xq_small = xq[:q_small].copy()
    faiss.normalize_L2(xq_small)
    _, I_s = idx_small.search(xq_small, k_hub)

    # Filter out -1 (FAISS pads with -1 when k > nb or edge cases)
    I_flat = I_s.reshape(-1)
    I_flat = I_flat[I_flat >= 0].astype(np.int64)
    if I_flat.size == 0:
        print("Hubness@k: not enough data to compute (no valid neighbors).")
        return
    counts = np.bincount(I_flat, minlength=d_small)
    print(f"Hubness@{k_hub} (sample): mean={counts.mean():.2f}  p99={np.percentile(counts,99):.1f}")

# ----------------------------
# Index size proxy
# ----------------------------
def index_size_bytes(index):
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        path = fp.name
    try:
        faiss.write_index(index, path)
        return os.path.getsize(path)
    finally:
        try: os.remove(path)
        except Exception: pass

# ----------------------------
# CLI parsing
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="FAISS baseline: Flat vs IVF-Flat with L2 or cosine; supports synthetic or real embeddings."
    )
    p.add_argument("--mode", choices=["synthetic", "real"], help="Auto-infers if omitted.")
    p.add_argument("--metric", choices=["l2", "cosine"], default=DEFAULTS["metric"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    # synthetic knobs
    p.add_argument("--d", type=int, default=DEFAULTS["d"], help="Dimension (synthetic mode only)")
    p.add_argument("--nb", type=int, default=DEFAULTS["nb"], help="DB vectors (synthetic)")
    p.add_argument("--nq", type=int, default=DEFAULTS["nq"], help="Query vectors (synthetic)")
    # shared knobs
    p.add_argument("--k", type=int, default=DEFAULTS["k"])
    p.add_argument("--ivf-nlist", dest="ivf_nlist", type=int, default=DEFAULTS["ivf_nlist"])
    p.add_argument("--train-size", dest="train_size", type=int, default=DEFAULTS["train_size"])
    p.add_argument("--nprobe", dest="nprobe_sweep", type=int, nargs="+", default=DEFAULTS["nprobe_sweep"],
                   help="Space-separated list, e.g. --nprobe 1 8 16 32")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=DEFAULTS["batch_size"])
    # real data inputs
    p.add_argument("--real-db-npy", dest="real_db_npy", type=str, help="Path to emb_db.npy")
    p.add_argument("--real-q-npy", dest="real_q_npy", type=str, help="Path to emb_q.npy")
    p.add_argument("--texts-db", dest="texts_db_path", type=str, help="Path to db.txt (one doc per line)")
    p.add_argument("--texts-q",  dest="texts_q_path",  type=str, help="Path to q.txt  (one query per line)")
    p.add_argument("--embed-model", dest="embed_model", type=str, default=DEFAULTS["embed_model"])
    return p

def make_cfg_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = DEFAULTS.copy()
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    # Infer mode if not provided
    if cfg["mode"] is None:
        if (cfg["real_db_npy"] and cfg["real_q_npy"]) or (cfg["texts_db_path"] and cfg["texts_q_path"]):
            cfg["mode"] = "real"
        else:
            cfg["mode"] = "synthetic"
    # Light validation
    if cfg["mode"] == "real":
        have_npy = bool(cfg["real_db_npy"] and cfg["real_q_npy"])
        have_txt = bool(cfg["texts_db_path"] and cfg["texts_q_path"])
        if have_npy and have_txt:
            raise SystemExit("Pass EITHER .npy inputs OR text inputs, not both.")
        if not (have_npy or have_txt):
            raise SystemExit("For real mode, provide --real-db-npy/--real-q-npy OR --texts-db/--texts-q.")
    return cfg

# ----------------------------
# Main
# ----------------------------
def main(cfg):
    np.random.seed(cfg["seed"])
    xb, xq = get_data(cfg)
    print(f"Data: xb={xb.shape}, xq={xq.shape}")

    # --- Effective k (cannot exceed DB size)
    k_eff = min(int(cfg["k"]), int(xb.shape[0]))
    if k_eff != cfg["k"]:
        print(f"Note: requested k={cfg['k']} > db size={xb.shape[0]} → using k={k_eff}")

    # --- Normalize if cosine
    xb, xq = maybe_normalize_for_cosine(xb, xq, cfg["metric"])

    # --- Diagnostics for real embeddings (robust to tiny datasets)
    if cfg["mode"] == "real":
        print("\n[Diagnostics for real embeddings]")
        print_embedding_stats(xb.copy(), xq.copy())

    # --- Ground truth with Flat
    print("\nComputing ground truth with Flat…")
    gtD, gtI = ground_truth_flat(xb, xq, k_eff, cfg["metric"])

    # --- Flat search (exact baseline)
    print("\nFlat search…")
    flat = build_flat(xb, cfg["metric"])
    flat_sz = index_size_bytes(flat)
    resF = search_batched(flat, xq, k_eff, cfg["batch_size"])
    r1F, rkF = recall_at_k(resF["I"], gtI, k_eff)

    # --- IVF,Flat build: clamp nlist to available training points
    print("\nIVF,Flat build/search…")
    N = xb.shape[0]
    train_size_eff = max(1, min(int(cfg["train_size"]), N))
    nlist_req = int(cfg["ivf_nlist"])
    nlist_eff = max(1, min(nlist_req, train_size_eff))
    if nlist_eff != nlist_req:
        print(f"Note: ivf_nlist={nlist_req} > train_size={train_size_eff} → using nlist={nlist_eff}")

    t0 = time.perf_counter()
    ivf = build_ivf_flat(xb, cfg["metric"], nlist_eff, train_size_eff, cfg["seed"])
    build_ivf = time.perf_counter() - t0
    ivf_sz = index_size_bytes(ivf)

    # --- Results table
    header = ["Index", "nprobe", "nlist", "Size", "P50 ms", "P95 ms", "R@1", f"R@{k_eff}"]
    print("\n" + " | ".join(header))
    print(" | ".join("-" * len(h) for h in header))

    # Flat row
    print(" | ".join([
        "Flat", "-", "-",
        f"{flat_sz/1024/1024:.1f} MB",
        f"{resF['p50_ms']:.2f}", f"{resF['p95_ms']:.2f}",
        f"{r1F:.3f}", f"{rkF:.3f}"
    ]))

    # IVF rows (clamp nprobe ≤ nlist)
    for nprobe in cfg["nprobe_sweep"]:
        nprobe_eff = int(min(int(nprobe), nlist_eff))
        if nprobe_eff != nprobe:
            print(f"(note) clamping nprobe {nprobe} → {nprobe_eff} (nlist={nlist_eff})")
        ivf.nprobe = nprobe_eff
        res = search_batched(ivf, xq, k_eff, cfg["batch_size"])
        r1, rk = recall_at_k(res["I"], gtI, k_eff)
        print(" | ".join([
            "IVF,Flat", str(nprobe_eff), str(nlist_eff),
            f"{ivf_sz/1024/1024:.1f} MB",
            f"{res['p50_ms']:.2f}", f"{res['p95_ms']:.2f}",
            f"{r1:.3f}", f"{rk:.3f}"
        ]))

    print(f"\nIVF build time: {build_ivf:.2f}s")



if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    cfg = make_cfg_from_args(args)
    main(cfg)
