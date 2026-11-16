import ollama
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "qwen3:4b"

DOCUMENT_PATH = Path("data/alice.txt")

# --------------------------------------------------
# 2. Load the document
# --------------------------------------------------

def load_document(file_path):
    """Read a text document from disk."""

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")

    print(f"Document loaded successfully: {file_path.name}")
    print(f"Characters: {len(text):,}")

    return text

from pathlib import Path

DOCUMENT_PATH = Path("data/alice.txt")

document = load_document(DOCUMENT_PATH)

print(document[:500])



# --------------------------------------------------
# 3. Split document into chunks
# --------------------------------------------------

def split_into_chunks(text, chunk_size=1000):
    """Split text into smaller chunks."""

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            current_chunk = paragraph + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    print(f"Created {len(chunks)} chunks.")

    return chunks


# Load document
document = load_document(DOCUMENT_PATH)

# Split into chunks
chunks = split_into_chunks(document)

    # Total number of chunks
print("Total chunks:", len(chunks))

# Display the length of each chunk
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i + 1}: {len(chunk)} characters")


# --------------------------------------------------
# 4. Generate embeddings using Nomic
# --------------------------------------------------
# example:
# chunks = [
#     "Apple is a technology company.",
#     "Microsoft develops software.",
#     "Google develops AI technologies."
# ]
chunks = [
    "Apple.",
    "banana",
    "computer.",
    "computers."
]
def generate_embeddings(chunks):
    """Convert document chunks into numerical vectors."""

    print("\nGenerating embeddings...")

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunks
    )

    embeddings = np.array(response["embeddings"])

    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Vector dimensions: {embeddings.shape[1]}")
    print(f"Vector dimensions: {embeddings}")
    print(f"Vector Whole dimensions: {embeddings.shape}")

    return embeddings

def generate_embeddings(chunks):

    print("\nGenerating embeddings...")

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunks
    )

    embeddings = np.array(response["embeddings"])

    print("\nEmbedding Results")
    print("-----------------")

    print("Number of chunks:", len(chunks))
    print("Number of embeddings:", len(embeddings))
    print("Vector dimensions:", embeddings.shape[1])

    # Display the first chunk
    print("\nFirst chunk:")
    print(chunks[0])

    # Display its embedding
    print("\nFirst embedding (first 10 values):")
    print(embeddings[0][:10])

    return embeddings


# Generate embeddings
embeddings = generate_embeddings(chunks)

# Example for Testing Cosine similarity

import numpy as np

A = np.array([1, 2, 3])
B = np.array([2, 4, 6])

# 1. Cosine similarity
cosine = np.dot(A, B) / (
    np.linalg.norm(A) * np.linalg.norm(B)
)

# 2. Euclidean distance
euclidean = np.linalg.norm(A - B)

# 3. Dot product
dot_product = np.dot(A, B)

print("Cosine similarity:", cosine)
print("Euclidean distance:", euclidean)
print("Dot product:", dot_product)