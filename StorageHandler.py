import os
import faiss
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import pandas as pd
from tqdm import tqdm
import numpy as np

# Configure the Generative AI API key
# This enables communication with the Generative AI embedding model
genai.configure(api_key="AIzaSyChjMzI1Pj2EhhJysR4hSI9IvCB1Aa1k8M")

# Similarity threshold for FAISS queries
threshold = 0.8

# Function to load text files from a specified directory
def load_text_files(directory):
    """
    Load and return the content of all .txt files in the specified directory.

    Args:
        directory (str): Path to the directory containing text files.

    Returns:
        list: A list of documents loaded from the text files.
    """
    documents = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            try:
                # Attempt to load using UTF-8 encoding
                loader = TextLoader(os.path.join(directory, file_name), encoding='utf-8')
                documents.extend(loader.load())
            except UnicodeDecodeError:
                print(f"Error decoding file: {file_name}, trying cp1252 encoding.")
                loader = TextLoader(os.path.join(directory, file_name), encoding='cp1252')
                documents.extend(loader.load())
    return documents

# Function to embed and store documents in a FAISS index
def embed_and_store_documents(text_dir, output_file='document_dataset.csv'):
    """
    Process text files, embed their content, and store them in a FAISS index and a CSV file.

    Args:
        text_dir (str): Directory containing the text files to process.
        output_file (str): Path to save the resulting metadata and content as a CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing metadata and chunk content.
    """
    # Load and split documents into smaller chunks
    documents = load_text_files(text_dir)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Embed each chunk with a loading bar
    chunk_embeddings = np.array([
        query_function(chunk.page_content)
        for chunk in tqdm(chunks, desc="Embedding chunks", unit="chunk", total=len(chunks))
    ])

    # Prepare metadata for the chunks
    metadata = [{"file_name": chunk.metadata.get("source", "unknown")} for chunk in chunks]

    # Print when the embedding process is complete
    print("Embedding process complete.")

    # Initialize a FAISS index using L2 distance
    dim = chunk_embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(dim)
    index.add(chunk_embeddings.astype(np.float32))  # Add embeddings to the index

    # Store metadata and chunk content in a DataFrame
    df = pd.DataFrame({
        'metadata': metadata,
        'chunk_content': [chunk.page_content for chunk in chunks]
    })

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)

    # Save the FAISS index to a file
    faiss.write_index(index, 'faiss_index.index')

    print(f"Data saved to {output_file}. FAISS index saved as 'faiss_index.index'.")

    return df

# Function to generate embeddings for a given text
def query_function(txt):
    """
    Generate embeddings for the given text using Generative AI's embedding model.

    Args:
        txt (str): The text content to embed.

    Returns:
        numpy.ndarray: The embedding vector for the text.
    """
    embedding_response = genai.embed_content(
        model="models/text-embedding-004",  # Model for text embeddings
        content=txt
    )
    return np.array(embedding_response['embedding'])

# Function to query the FAISS index
def query_faiss(index, query_text, metadata, chunks, k=5):
    """
    Perform a similarity search in the FAISS index for the given query text.

    Args:
        index (faiss.Index): The FAISS index.
        query_text (str): The text query.
        metadata (list): Metadata associated with the chunks.
        chunks (list): Content of the chunks.
        k (int): Number of top results to return.

    Returns:
        tuple: The retrieved chunk text and metadata.
    """
    # Generate an embedding for the query text
    query_embedding = np.array([query_function(query_text)])

    # Search the FAISS index
    D, I = index.search(query_embedding.astype(np.float32), k)
    closest_distance = D[0][0]

    # If the closest distance exceeds the threshold, return a default message
    if closest_distance > threshold:
        return "//insouciant knowledge//", metadata[I[0][0]]['file_name']

    rMeta = ""
    rTxt = ""
    for i in range(k):
        # Append metadata and content for the top results
        rMeta += f"{metadata[I[0][i]]['file_name']}\n"
        rTxt += f"{chunks[I[0][i]]}\n{'-'*30}\n"

    return rTxt, rMeta

# Main execution point
if __name__ == "__main__":
    # Embed and store text files from the specified directory
    text_dir = "readTextFiles"
    embed_and_store_documents(text_dir)
