## Docs

```bash
sudo setup.sh
```

```bash
python make_embeddings_npy.py synthetic --N 200000 --Q 2000 --d 768 --normalize --out_dir .
```

```bash
# Prepare files with one item per line
printf "First doc\nSecond doc\n" > ./txt/db.txt
printf "First doc\nAnother query\n" > ./txt/q.txt
```

```bash
python - <<'PY'
> import torch
> print("cuda_available:", torch.cuda.is_available())
> print("torch:", torch.__version__, "cuda:", torch.version.cuda)
> if torch.cuda.is_available():
>     print("gpu:", torch.cuda.get_device_name(0))
> PY
```

```bash
PYTHONNOUSERSITE=1  python src/make_embeddings_npy.py text   --db_txt ./txt/db.txt --q_txt ./txt/q.txt   --model all-MiniLM-L6-v2 --batch_size 512 --normalize --out_dir ./embeddings/

```

```bash
PYTHONNOUSERSITE=1 python src/make_embeddings_npy.py synthetic \
  --N 200000 --Q 2000 --d 384 --normalize --seed 123 \
  --out_dir ./embeddings/
```

```bash
PYTHONNOUSERSITE=1  python src/faiss_step1_cosine_or_synthetic.py --mode synthetic --d 128 --nb 200000 --nq 2000 --metric l2 --nprobe 1 8 16 32

PYTHONNOUSERSITE=1   python src/faiss_step1_cosine_or_synthetic.py --mode real --metric cosine \
  --real-db-npy ./embeddings/emb_db.npy --real-q-npy ./embeddings/emb_q.npy \
  --ivf-nlist 4096 --nprobe 1 8 16 32

PYTHONNOUSERSITE=1    python src/faiss_step1_cosine_or_synthetic.py --metric cosine \
  --texts-db ./txt/db.txt --texts-q ./txt/q.txt --embed-model all-MiniLM-L6-v2 \
  --ivf-nlist 8192 --nprobe 8 16 32
  
```