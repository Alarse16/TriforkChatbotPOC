from StorageHandler import query_faiss
import faiss
import pandas as pd
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import os
import json
import subprocess
from ImageHandler import read_images
from StorageHandler import embed_and_store_documents
import google.generativeai as genai

genai.configure(api_key="AIzaSyChjMzI1Pj2EhhJysR4hSI9IvCB1Aa1k8M")

# Function to load the FAISS index, metadata, and chunks from files
def load_index_and_data(index_file='faiss_index.index', data_file='document_dataset.csv'):
    # Load the FAISS index
    index = faiss.read_index(index_file)
    print("FAISS index loaded successfully.")

    # Load the dataset (CSV) containing metadata and chunks
    df = pd.read_csv(data_file)
    print(f"Dataset loaded from {data_file}.")

    # Extract metadata and chunk content
    metadata = df['metadata'].apply(eval).tolist()  # Convert the string representation of dict back to actual dict
    chunks = df['chunk_content'].tolist()

    return index, metadata, chunks

if os.path.exists("config.json"):
    index, metadata, chunks = load_index_and_data()

template = """
Answer this question concisely, short and precisely: {question}
based ONLY the information given in this chunk of text and its meta data: {knowledge}, {meta}.
When giving an answer, ALWAYS include the name of the file of the chunk described in the meta data where you got the relevant information.
Here is the conversation history: {context}
"""

model = genai.GenerativeModel("gemini-1.5-flash")
# prompt = ChatPromptTemplate.from_template(template)
# chain = prompt | model


def llm_answer_to_promt(context, question, knowledge = ""):
    knowledge, meta = query_faiss(index, question, metadata, chunks)

    if knowledge == "//insouciant knowledge//":
        standartNoAnswer = "I do not have sufficient information to answer this question."
        return standartNoAnswer
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

        # Find the position of the keyword
        position = response.find("text") + 6
        result = response[position:]

        # Find the position of the first closing curly brace and slice the string
        end_position = result.find("\\n")  # Include the closing brace
        result = result[:end_position]

        return result


def send_message(event=None):
    user_message = input_box.get()
    context = ""
    if user_message.strip().lower() == "exit":
        root.destroy()
        return

    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"You: {user_message}\n")

    # Force UI update to show the user's message
    root.update_idletasks()

    response = f"BOT: {llm_answer_to_promt(context, user_message)}\n{'-'*100}\n"
    context += f"\nUser: {user_message}\nAI: {response}"
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
chat_window.insert(tk.END,
                   "I am knowledgeable about everything related to the PS2 \"God of War\" manual, and nothing else. Feel free to ask me anything. If I do not have the information you are seeking, I will let you know. \n")

# Force UI update to show the user's message
root.update_idletasks()

# Input box
input_box = tk.Entry(root, width=100)
input_box.pack(padx=10, pady=5)
input_box.bind("<Return>", send_message)

# Send button
send_button = tk.Button(root, text="Ask question", command=send_message)
send_button.pack(pady=5)


# First Run ----------------------------------------
def is_first_run(config_file="config.json"):
    if not os.path.exists(config_file):
        return True
    with open(config_file, "r") as file:
        config = json.load(file)
    return config.get("first_run", True)


def mark_first_run_complete(config_file="config.json"):
    with open(config_file, "w") as file:
        json.dump({"first_run": False}, file)



def run_initial_setup():
    print("Running initial setup...")
    # Add any other setup tasks here
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

    # Main program logic here
    # Run the application
    root.mainloop()


if __name__ == "__main__":
    main()
