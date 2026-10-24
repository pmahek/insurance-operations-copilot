
from sentence_transformers import SentenceTransformer

# Step 1: Load a pretrained embedding model.
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Create sample insurance text.
documents = [
    "The policyholder must notify the insurer within 24 hours of hospitalisation.",
    "Customers must submit identity proof and address proof for KYC.",
    "Motor insurance claims require the relevant claim documents.",
]

# Step 3: Convert all document sentences into embeddings.
document_embeddings = model.encode(documents)

# Step 4: Create an embedding for the employee's question.
query = "How soon must a customer report hospital admission?"
query_embedding = model.encode(query)

# Step 5: Inspect the results.
print("Number of documents:", len(documents))
print("Document embeddings shape:", document_embeddings.shape)
print("Query embedding shape:", query_embedding.shape)

print("\nFirst document:")
print(documents[0])

print("\nFirst document embedding:")
print(document_embeddings[0])

print("\nQuestion:")
print(query)

print("\nQuestion embedding:")
print(query_embedding)
