from StorageHandler import query_faiss
import faiss
import pandas as pd
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import os
import json
from ImageHandler import read_images
from StorageHandler import embed_and_store_documents
import google.generativeai as genai

# Configure the GenAI API key
# This enables communication with the Generative AI model.
genai.configure(api_key="AIzaSyChjMzI1Pj2EhhJysR4hSI9IvCB1Aa1k8M")

# Function to load the FAISS index, metadata, and chunks from files
def load_index_and_data(index_file='faiss_index.index', data_file='document_dataset.csv'):
    """
    Load the FAISS index and dataset containing metadata and chunks.

    Args:
        index_file (str): Path to the FAISS index file.
        data_file (str): Path to the dataset CSV file.

    Returns:
        tuple: FAISS index, metadata, and chunks.
    """
    # Load the FAISS index
    index = faiss.read_index(index_file)
    print("FAISS index loaded successfully.")

    # Load the dataset (CSV) containing metadata and chunks
    df = pd.read_csv(data_file)
    print(f"Dataset loaded from {data_file}.")

    # Extract metadata and chunk content
    metadata = df['metadata'].apply(eval).tolist()  # Convert string representation of dict to dict
    chunks = df['chunk_content'].tolist()

    return index, metadata, chunks

# Check if the configuration file exists and load the index and data
if os.path.exists("config.json"):
    index, metadata, chunks = load_index_and_data()

# Template for the question-answering prompt
template = """
Answer this question concisely, short and precisely: {question}
based ONLY the information given in this chunk of text and its meta data: {knowledge}, {meta}.
When giving an answer, ALWAYS include the name of the file of the chunk described in the meta data where you got the relevant information.
Here is the conversation history: {context}
"""

# Initialize the Generative AI model
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to generate answers from the model
def llm_answer_to_promt(context, question, knowledge=""):
    """
    Generate an answer to a question based on context, metadata, and knowledge.

    Args:
        context (str): Conversation history.
        question (str): User's question.
        knowledge (str): Knowledge to base the answer on (optional).

    Returns:
        str: Generated response or standard no-answer message.
    """
    knowledge, meta = query_faiss(index, question, metadata, chunks)

    if knowledge == "//insouciant knowledge//":
        # Return a standard message if no knowledge is found
        return "I do not have sufficient information to answer this question."
    else:
        # Format the template with the parameters
        prompt = template.format(
            question=question,
            knowledge=knowledge,
            meta=meta,
            context=context
        )

        # Use the model to generate content
        response = str(model.generate_content(prompt))

        # Extract the relevant part of the response
        position = response.find("text") + 6
        result = response[position:]
        end_position = result.find("\\n")
        result = result[:end_position]

        return result

# Function to handle user input and generate responses
def send_message(event=None):
    """
    Handle sending a message in the chat interface.

    Args:
        event: Event triggered by the user (optional).
    """
    user_message = input_box.get()
    context = ""
    if user_message.strip().lower() == "exit":
        root.destroy()
        return

    # Display the user's message in the chat window
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"You: {user_message}\n")
    root.update_idletasks()

    # Generate a response from the model
    response = f"BOT: {llm_answer_to_promt(context, user_message)}\n{'-'*100}\n"
    context += f"\nUser: {user_message}\nAI: {response}"

    # Display the response in the chat window
    chat_window.insert(tk.END, response + "\n\n")
    chat_window.config(state=tk.DISABLED)
    input_box.delete(0, tk.END)
    chat_window.see(tk.END)

# Create the main application window
root = tk.Tk()
root.title("Kratos' Wisdom")

# Chat display area
chat_window = ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, height=30, width=100)
chat_window.pack(padx=20, pady=10)
chat_window.config(state=tk.NORMAL)
chat_window.insert(tk.END, "I am knowledgeable about everything related to the PS2 'God of War' manual, and nothing else. Feel free to ask me anything. If I do not have the information you are seeking, I will let you know. \n")
root.update_idletasks()

# Input box
input_box = tk.Entry(root, width=100)
input_box.pack(padx=10, pady=5)
input_box.bind("<Return>", send_message)

# Send button
send_button = tk.Button(root, text="Ask question", command=send_message)
send_button.pack(pady=5)

# First Run Functions

def is_first_run(config_file="config.json"):
    """
    Check if the application is running for the first time.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        bool: True if it is the first run, False otherwise.
    """
    if not os.path.exists(config_file):
        return True
    with open(config_file, "r") as file:
        config = json.load(file)
    return config.get("first_run", True)

def mark_first_run_complete(config_file="config.json"):
    """
    Mark the first run as complete by updating the configuration file.

    Args:
        config_file (str): Path to the configuration file.
    """
    with open(config_file, "w") as file:
        json.dump({"first_run": False}, file)

def run_initial_setup():
    """
    Perform the initial setup tasks such as reading images and embedding documents.
    """
    print("Running initial setup...")
    read_images()
    embed_and_store_documents("readTextFiles")
    global index, metadata, chunks
    index, metadata, chunks = load_index_and_data()

def main():
    config_file = "config.json"
    if is_first_run(config_file):
        print("First run detected. Running setup...")
        run_initial_setup()
        mark_first_run_complete(config_file)
    else:
        print("Not the first run. Skipping setup.")

    # Run the application
    root.mainloop()

if __name__ == "__main__":
    main()
