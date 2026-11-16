
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


# --------------------------------------------------
# 4. Generate embeddings using Nomic
# --------------------------------------------------

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

    return embeddings


# --------------------------------------------------
# 5. Retrieve relevant chunks
# --------------------------------------------------

def retrieve_chunks(question, chunks, embeddings, top_k=3):
    """Find the most relevant chunks using cosine similarity."""

    query_response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question
    )

    query_embedding = np.array(
        query_response["embeddings"][0]
    )

    # Cosine similarity
    similarities = (
        embeddings @ query_embedding
        / (
            np.linalg.norm(embeddings, axis=1)
            * np.linalg.norm(query_embedding)
            + 1e-10
        )
    )

    best_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in best_indices:
        results.append({
            "text": chunks[index],
            "score": float(similarities[index])
        })

    return results


# --------------------------------------------------
# 6. Generate answer using Qwen
# --------------------------------------------------

def generate_answer(question, retrieved_chunks):
    """Generate an answer using retrieved document context."""

    context = "\n\n".join(
        item["text"] for item in retrieved_chunks
    )

    prompt = f"""
You are a question-answering assistant.

Answer the question using ONLY the information
provided in the context below.

Instructions:
1. Do not use your prior knowledge.
2. Do not invent information.
3. If the answer is not explicitly supported
   by the context, say:
   "I cannot find the answer in the provided context."
4. Keep your answer concise.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.generate(
        model=GENERATION_MODEL,
        prompt=prompt
    )

    return response["response"]


# --------------------------------------------------
# 7. Main application
# --------------------------------------------------

def main():

    print("Starting RAG application...")

    # Load document
    document = load_document(DOCUMENT_PATH)

    # Split into chunks
    chunks = split_into_chunks(document)

    # Generate embeddings
    # For your RAG project, we generally generate embeddings for document chunks rather than individual words.
    embeddings = generate_embeddings(chunks)

    # Ask a question
    question = "Who does Alice follow down the rabbit hole?"

    print(f"\nQuestion: {question}")

    # Retrieve relevant information
    retrieved = retrieve_chunks(
        question,
        chunks,
        embeddings
    )

    print("\nRetrieved chunks:")

    for result in retrieved:
        print(f"\nSimilarity: {result['score']:.4f}")
        print(result["text"][:300])

    # Generate answer
    print("\nGenerating answer using Qwen...")

    answer = generate_answer(question, retrieved)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
