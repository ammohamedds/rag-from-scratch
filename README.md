# Build RAG from Scratch with Python

A beginner-friendly **Retrieval-Augmented Generation (RAG)** project built from scratch using Python, Ollama, and locally running AI models.

The purpose of this project is to understand how RAG works internally, rather than relying on frameworks such as LangChain. Each component is implemented and tested independently to understand its functionality.

## 1. Project Overview

Retrieval-Augmented Generation (RAG) combines information retrieval with Large Language Models (LLMs).

Instead of asking an LLM to answer a question using only its pre-trained knowledge, RAG retrieves relevant information from an external document and provides it to the LLM as context.

The main workflow is:

1. Load a document.
2. Split the document into smaller chunks.
3. Generate embeddings for each chunk.
4. Convert the user's question into an embedding.
5. Calculate similarity between the question and document embeddings.
6. Retrieve the most relevant chunks.
7. Provide the retrieved information and question to the LLM.
8. Generate an answer based on the supplied context.

### RAG Architecture

```text
                 DOCUMENT
                    |
                    v
              Load Document
                    |
                    v
               Text Chunks
                    |
                    v
             Embedding Model
             (Nomic Embed)
                    |
                    v
            Document Embeddings
                    |
                    |
USER QUESTION       |
      |             |
      v             |
  Query Embedding   |
      |             |
      v             v
    Cosine Similarity
          |
          v
     Top-K Retrieval
          |
          v
   Relevant Text Chunks
          |
          v
    Prompt Construction
          |
          v
      LLM (Qwen 3)
          |
          v
    Generated Answer
```

## 2. Technologies and Models

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| VS Code | Development environment |
| Ollama | Running AI models locally |
| NumPy | Numerical operations and vector calculations |
| Nomic Embed Text | Generating text embeddings |
| Qwen3 4B | Generating answers |
| GitHub | Source code management and documentation |

### Local AI Models

The project uses two separate models:

**1. Embedding model: `nomic-embed-text`**

Converts text into numerical vectors that represent semantic information.

**2. Generation model: `qwen3:4b`**

Receives the retrieved document context and user question, then generates a natural-language answer.

The embedding model and generation model have different responsibilities.

## 3. Understanding the Main Components

### 3.1 Document Loading

The first step is loading the source document into Python.

For this experiment, the source document is:

*Alice's Adventures in Wonderland* by Lewis Carroll.

The document is loaded as text so that it can be processed by subsequent stages.

### 3.2 Text Chunking

A complete document can contain thousands of paragraphs and words.

Passing the entire document to an embedding model or LLM is not always practical.

Therefore, we divide the document into smaller pieces called chunks.

Example:

```text
Original Document
       |
       v
Paragraph 1
Paragraph 2
Paragraph 3
Paragraph 4
Paragraph 5
       |
       v
    Chunks
       |
       +---- Chunk 1: Paragraphs 1 and 2
       |
       +---- Chunk 2: Paragraphs 3 and 4
       |
       +---- Chunk 3: Paragraph 5
```

The current implementation uses paragraph-based chunking with a configurable target size.

Important considerations:

- Preserve paragraph boundaries where possible.
- Avoid creating unnecessarily large chunks.
- Keep related information together.
- Consider adding chunk overlap to preserve context between neighbouring chunks.

**Why is chunking important?**

The retrieval process works with individual chunks. If related information is split across different chunks, the system may retrieve only part of the information needed to answer a question.

### 3.3 Generating Embeddings

An embedding converts text into a numerical vector.

For example, consider these sentences:

```text
I love eating apples and bananas.
I enjoy eating fruits.
I drive my car to work.
```

Although the first two sentences use different words, their meanings are related.

An embedding model attempts to represent these semantic relationships numerically.

Illustrative vectors:

```text
Apple and banana:
[0.90, 0.80, 0.10]

Eating fruits:
[0.85, 0.75, 0.20]

Driving a car:
[0.10, 0.20, 0.90]
```

These numbers are illustrative, not actual Nomic model outputs.

The first two vectors have similar directions, indicating greater semantic similarity.

The actual model produces vectors with hundreds of dimensions.

In this project, each document chunk receives its own embedding.

```python
def generate_embeddings(chunks):

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunks
    )

    embeddings = np.array(response["embeddings"])

    return embeddings
```

For example, if we have five chunks and the model produces 768-dimensional embeddings, the resulting NumPy array has this shape:

```text
(5, 768)
```

This means:

- 5 document chunks.
- 768 numerical values per embedding.

The actual vector dimensions depend on the selected embedding model.

### 3.4 Query Embedding

When a user submits a question, we also convert that question into an embedding.

For example:

```text
Question:
Who does Alice follow down the rabbit hole?
```

The question is converted into a numerical vector using the same embedding model used for the document chunks.

This allows us to compare the question against the document embeddings.

### 3.5 Cosine Similarity

After generating document and query embeddings, we need a mathematical method to measure their similarity.

This project initially uses cosine similarity.

Cosine similarity measures the angle between two vectors rather than their absolute magnitudes.

Formula:

\[
\operatorname{Cosine}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

Where:

- A represents a document embedding.
- B represents the query embedding.
- A · B represents the dot product.
- The denominator represents the product of the two vector magnitudes.

Interpretation:

| Score | Meaning |
|---|---|
| Close to +1 | Similar vector directions |
| Close to 0 | Little directional similarity |
| Close to -1 | Opposite directions |

For text embeddings, higher cosine similarity generally indicates greater semantic similarity.

**Why cosine similarity?**

It is a useful starting point for semantic search because it compares vector directions rather than allowing vector magnitude alone to dominate the comparison.

Other similarity and distance methods include:

- Euclidean distance: measures the straight-line distance between vectors.
- Dot product: measures the product of corresponding vector values.

The appropriate metric depends on the embedding model and its training objectives.

### 3.6 Top-K Retrieval

After calculating cosine similarity, we need to identify the most relevant document chunks.

For this, we use NumPy's `argsort()`.

```python
best_indices = np.argsort(similarities)[::-1][:top_k]

results = []

for index in best_indices:
    results.append({
        "text": chunks[index],
        "score": float(similarities[index])
    })

return results
```

Explanation:

- `np.argsort()` returns indices that would sort the similarity scores in ascending order.
- `[::-1]` reverses the order so that the highest scores come first.
- `[:top_k]` selects the required number of results.
- The loop retrieves the original text and corresponding similarity score.

For example:

```python
similarities = [0.85, 0.12, 0.35, 0.91, 0.08]

top_k = 2
```

The selected indices are:

```python
[3, 0]
```

The two highest-scoring chunks are therefore retrieved first.

**Important:** Similarity measures semantic relevance, not whether a chunk contains the exact answer. A high similarity score does not guarantee that the retrieved text answers the question.

## 4. Prompting and Answer Generation

Once the relevant chunks have been retrieved, we pass them to the generation model.

This process is called prompt construction.

A prompt provides the LLM with:

1. Instructions about how to answer.
2. Retrieved document context.
3. The user's question.

### Example Prompt

```text
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
[Retrieved document chunks]

Question:
Who does Alice follow down the rabbit hole?

Answer:
```

### Why is prompting important?

An LLM such as Qwen already has knowledge learned during its training.

Without appropriate instructions, it may answer using its pre-trained knowledge, even when the retrieved document context does not contain the answer.

The purpose of grounding instructions is to encourage the model to answer using the retrieved information.

However, prompting alone cannot guarantee that the model will never use prior knowledge or generate unsupported information.

This is why retrieval quality and answer quality must be evaluated separately.

## 5. Practical Experiment: Alice in Wonderland

This experiment demonstrates the complete relationship between retrieval, similarity scores and answer generation.

### Question

```text
Who does Alice follow down the rabbit hole?
```

### Retrieved Chunks

The retrieval stage returned the following results:

**Result 1**

Similarity: `0.7369`

```text
CHAPTER I.     Down the Rabbit-Hole
CHAPTER II.    The Pool of Tears
CHAPTER III.   A Caucus-Race and a Long Tale
CHAPTER IV.    The Rabbit Sends in a Little Bill
CHAPTER V.     Advice from a Caterpillar
CHAPTER VI.    Pig and Pepper
CHAPTER VII.   A Mad Tea-Party
CHAPTER VIII.  The Queen's Croquet-Ground
```

**Result 2**

Similarity: `0.7252`

```text
The rabbit-hole went straight on like a tunnel for some way,
and then dipped suddenly down, so suddenly that Alice had not
a moment to think about stopping herself before she found
herself falling down a very deep well.
```

**Result 3**

Similarity: `0.6845`

```text
It was the White Rabbit, trotting slowly back again, and looking
anxiously about as it went, as if it had lost something; and she
heard it muttering to itself "The Duchess! The Duchess! Oh my
dear paws! Oh my fur and whiskers!"
```

*Note: The displayed passages are shortened excerpts of the retrieved chunks.*

### Analysis of Retrieval Results

The results demonstrate an important limitation of semantic retrieval.

| Rank | Similarity | Observation |
|---|---:|---|
| 1 | 0.7369 | Chapter headings are related to the topic but do not contain the answer. |
| 2 | 0.7252 | Describes Alice falling down the rabbit hole but does not identify who she follows. |
| 3 | 0.6845 | Mentions the White Rabbit but does not explicitly establish that Alice follows it down the hole. |

The highest similarity score does not necessarily indicate the most useful passage for answering the question.

Cosine similarity measures the relationship between the question and retrieved text embeddings. It does not verify whether the answer is explicitly present.

### Generated Answer Using Qwen

The original generation prompt produced this answer:

```text
Based on the supplied document context, I can find the
answer to the question "Who does Alice follow down the
rabbit hole?"

In the context, it states: "It was the White Rabbit,
trotting slowly back again, and looking anxiously about
as it went, as if it had lost something..."

This passage shows that Alice saw the White Rabbit near
the rabbit hole, and the context implies that she followed
this rabbit down the hole.

In the classic story, Alice follows the White Rabbit down
the rabbit hole, and the context confirms that the White
Rabbit was the one she encountered near the entrance.

Therefore, the answer is: The White Rabbit.
```

### What does this result tell us?

The answer is factually correct, but the retrieved chunks do not explicitly establish the complete answer.

There are two possible sources of information:

**1. Retrieved context**

Qwen can use information from the supplied passages to form an answer.

**2. Pre-trained knowledge**

Qwen may already know the story *Alice's Adventures in Wonderland* from its original training data.

Therefore, it may generate the correct answer even when retrieval does not provide sufficient evidence.

We cannot determine precisely which source Qwen relied on from the generated answer alone.

This is an important distinction when evaluating RAG systems.

A correct answer does not necessarily mean that the retrieval process worked correctly.

## 6. Improving the Experiment with Strict Prompting

To investigate this behaviour, we can change the prompt to explicitly restrict answers to the supplied context.

For example:

```text
You are a question-answering assistant.

Answer ONLY using information explicitly supported
by the supplied context.

Do not use your prior knowledge.
Do not make assumptions or infer missing information.

If the context does not explicitly contain the answer,
respond with:

"I cannot find the answer in the provided context."

Keep your answer concise.
```

Using the same three retrieved chunks, the expected response should be:

```text
I cannot find the answer in the provided context.
```

Why?

Because the retrieved text does not explicitly establish that Alice followed the White Rabbit down the rabbit hole.

If Qwen still answers "The White Rabbit", it demonstrates that the model is not reliably following the grounding instruction.

Prompting helps control model behaviour, but it is not a guarantee against unsupported answers.

## 7. Lessons Learned

This experiment highlights several important concepts:

**1. Retrieval quality matters**

The embedding model may retrieve passages that are semantically related to a question but do not contain the required answer.

**2. Chunking affects retrieval**

If related information is split across different chunks, the system may retrieve only part of the relevant information.

**3. Similarity is not answer correctness**

A high similarity score does not mean that a passage contains the answer.

**4. LLMs have pre-trained knowledge**

An LLM can answer questions using information learned during training, independently of the retrieved document context.

**5. Prompting helps ground generation**

Clear instructions encourage the model to use the retrieved context and acknowledge when information is missing.

**6. Retrieval and generation should be evaluated separately**

A reliable RAG system needs both effective retrieval and accurate, evidence-based answer generation.

## 8. Current Limitations and Future Improvements

The project is intentionally being developed incrementally.

| Area | Current implementation | Potential improvement |
|---|---|---|
| Chunking | Paragraph-based chunking | Add chunk overlap and handle oversized paragraphs |
| Embeddings | Nomic Embed Text | Experiment with alternative embedding models |
| Similarity | Cosine similarity | Compare cosine similarity, Euclidean distance and dot product |
| Retrieval | Top-K selection | Introduce similarity thresholds |
| Prompting | Context-based prompting | Improve grounding and missing-answer handling |
| Evaluation | Manual experiments | Evaluate retrieval accuracy and answer accuracy separately |
| Storage | In-memory NumPy arrays | Introduce persistent vector storage later |

These improvements will be introduced gradually as the underlying concepts become clear.

## 9. Project Learning Objectives

The main objective is to understand how to build a RAG system from scratch.

By completing this project, the learning objectives are:

- Understand document loading and text preprocessing.
- Understand chunking and its impact on retrieval.
- Understand embeddings and vector representations.
- Understand cosine similarity and other vector comparison methods.
- Implement semantic search using NumPy.
- Understand Top-K retrieval.
- Understand prompt construction and context grounding.
- Connect local embedding and generation models using Ollama.
- Evaluate the quality of retrieval and generated answers.

## 10. Running the Project

### Prerequisites

- Python installed.
- VS Code installed.
- Ollama installed and running.

### Download the Models

```powershell
ollama pull nomic-embed-text
ollama pull qwen3:4b
```

### Install Python Dependencies

```powershell
python -m pip install ollama numpy
```

### Run the Application

```powershell
python main.py
```

*The command assumes your application entry point is named `rag.py`.*

---

## Conclusion

This project demonstrates the fundamental stages of a Retrieval-Augmented Generation system using locally running AI models.

Rather than relying on pre-built RAG frameworks, the objective is to implement each component independently and understand the underlying concepts.

The Alice in Wonderland experiment demonstrates why successful RAG applications require more than an embedding model and a language model: they also require effective retrieval, appropriate chunking, well-designed prompts and reliable evaluation.

The project will continue to evolve incrementally, with a focus on understanding each component before introducing additional technologies or complexity.