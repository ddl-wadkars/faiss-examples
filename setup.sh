pip install -U pip

python -m pip install -U pip "numpy<2"

# install PyTorch for your CUDA (check `nvidia-smi` → e.g., CUDA Version: 12.1)
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# remove conflicting extras if you installed them before
#python -m pip uninstall -y transformer-engine flash-attn xformers
pip install -U "transformers>=4.41,<5" "sentence-transformers>=2.6"
# now install sentence-transformers
#python -m pip install "sentence-transformers>=2.2"

pip install faiss-cpu
pip install faiss-gpu
