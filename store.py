from sentence_transformers import SentenceTransformer
import json
import os
import faiss
import numpy as np
from tqdm import tqdm


with open("data.json", "r") as f:
    data = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = []
metadata = []

def dict_to_text(disease_dict):
    return "\n".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in disease_dict.items())


for dict in data:
     text = dict_to_text(dict)
     texts.append(text)


# print(texts)
     


embedding_dim = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)


## embedding texts

for text in texts:
    embeddings = model.encode(texts, show_progress_bar=True)



i = 1


## storing in vector database


for texts in tqdm(texts):

    emb = embeddings[i-1]
    index.add(np.array([emb], dtype=np.float32))

    metadata.append({
        f"{i}": texts
    })

    i += 1


print(f"✅ Indexed {len(metadata)} texts.")

faiss.write_index(index, "information.index")

with open("metadata.json", "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
