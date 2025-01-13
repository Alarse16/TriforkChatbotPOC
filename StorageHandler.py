import os
import faiss
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import pandas as pd
from tqdm import tqdm

genai.configure(api_key="AIzaSyChjMzI1Pj2EhhJysR4hSI9IvCB1Aa1k8M")
threshold = 0.8

# Function to load text files
def load_text_files(directory):
    documents = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            try:
                loader = TextLoader(os.path.join(directory, file_name), encoding='utf-8')
                documents.extend(loader.load())
            except UnicodeDecodeError:
                print(f"Error decoding file: {file_name}, trying cp1252 encoding.")
                loader = TextLoader(os.path.join(directory, file_name), encoding='cp1252')
                documents.extend(loader.load())
    return documents


# Function to embed and store documents in FAISS
def embed_and_store_documents(text_dir, output_file='document_dataset.csv'):
    # Load and split documents into chunks
    documents = load_text_files(text_dir)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Embed the chunks with a loading bar
    chunk_embeddings = np.array([
        query_function(chunk.page_content)
        for chunk in tqdm(chunks, desc="Embedding chunks", unit="chunk", total=len(chunks))
    ])

    # Prepare metadata (e.g., file names)
    metadata = [{"file_name": chunk.metadata.get("source", "unknown")} for chunk in chunks]

    # Print when processing is complete
    print("Embedding process complete.")

    # Initialize FAISS index
    dim = chunk_embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(dim)  # FAISS L2 distance index
    index.add(chunk_embeddings.astype(np.float32))  # Add embeddings to the index

    # Store embeddings, metadata, and chunks in a pandas DataFrame
    df = pd.DataFrame({
        'metadata': metadata,
        'chunk_content': [chunk.page_content for chunk in chunks]
    })

    # Save the DataFrame to a CSV file (or other formats like Parquet, if preferred)
    df.to_csv(output_file, index=False)

    # Optionally: Save the FAISS index (you can store it separately or serialize it)
    faiss.write_index(index, 'faiss_index.index')

    print(f"Data saved to {output_file}. FAISS index saved as 'faiss_index.index'.")

    return df  # You can return the DataFrame if you want to work with it after saving

def query_function(txt):
    # Return the embeddings using Google Generative AI's embedding model
    embedding_response = genai.embed_content(
        model="models/text-embedding-004",  # Assuming this is the correct model name
        content=txt  # Single chunk of text
    )
    return np.array(embedding_response['embedding'])  # Ensure the embedding is returned as a NumPy array

import numpy as np

# Function to query the FAISS index
def query_faiss(index, query_text, metadata, chunks, k=5):
    # Generate embedding for the query text
    query_embedding = np.array([query_function(query_text)])

    # Perform similarity search in FAISS index
    D, I = index.search(query_embedding.astype(np.float32), k)  # D = distances, I = indices

    closest_distance = D[0][0]
    # If closest distance is above threshold, return a default message
    if closest_distance > threshold:
        return "//insouciant knowledge//", metadata[I[0][0]]['file_name']

    rMeta = ""
    rTxt = ""
    for i in range(k):
        # Extract the metadata and content for the closest chunks
        rMeta += f"{metadata[I[0][i]]['file_name']}\n"
        rTxt += f"{chunks[I[0][i]]}\n{'-'*30}\n"

    return rTxt, rMeta

if __name__ == "__main__":
    # Preload the FAISS index
    text_dir = "readTextFiles"
    embed_and_store_documents(text_dir)
