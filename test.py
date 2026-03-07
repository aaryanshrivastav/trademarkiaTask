import torch
print(torch.__version__)
print(torch.rand(3,3))

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
print(model.encode("vector databases are cool"))