from sentence_transformers import SentenceTransformer, util

# Laad het voorgetrainde BERT model via SentenceTransformer
model = SentenceTransformer('GroNLP/bert-base-dutch-cased-gronings')

# Definieer de teksten waarvan je de gelijkenis wilt berekenen
text1 = "De zun schient vandoag."
text2 = "t Is n zunnege dag vandoag."

# Genereer embeddings voor beide teksten
embedding1 = model.encode(text1, convert_to_tensor=True)
embedding2 = model.encode(text2, convert_to_tensor=True)

# Bereken cosine similarity
cosine_similarity = util.cos_sim(embedding1, embedding2)

print(f"Cosine Similarity: {cosine_similarity.item():.4f}")