python -m venv ./venv 
source ./venv/bin/active
pip install -r requirements.txt
pip install faiss-gpu
ImportError: Could not import faiss python package. Please install it with `pip install faiss-gpu` (for CUDA supported GPU) or `pip install faiss-cpu` (depending on Python version).
pip install -U langchain-community
pip install sentence-transformers
docker exec -it ollama ollama run phi3
pip install -U langchain-huggingface

http://localhost:8000/any_question/playground/

http://localhost:8000/docs#/any_question/any_question_invoke_any_question_invoke_post



docker exec -it langchain ollama run phi3
