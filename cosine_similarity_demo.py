
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Create sample insurance passages
documents = [
    "The policyholder must notify the insurer within 24 hours of hospitalisation.",
    "Customers must submit identity proof and address proof for KYC.",
    "Motor insurance claims require the relevant claim documents."
]

# 3. Create the employee's question
query = "How soon must a customer report hospital admission?"

# 4. Convert passages and question into embeddings
document_embeddings = model.encode(documents)
query_embedding = model.encode([query])

# 5. Calculate similarity scores
scores = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]

# 6. Rank passages from most to least similar
ranked_results = sorted(
    zip(documents, scores),
    key=lambda item: item[1],
    reverse=True
)

# 7. Display the results
for rank, (document, score) in enumerate(ranked_results, start=1):
    print(f"\nRank: {rank}")
    print(f"Similarity: {score:.4f}")
    print(f"Passage: {document}")
