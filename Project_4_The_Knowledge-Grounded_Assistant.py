"""
Created by Mark Chmielecki for CS6313 - Foundations of Artificial Intelligence
Project 2 due Monday, September 28, 2026.   

RAG (Retrieval-Augmented Generation) utilities for document processing and querying.

First, the routine loads PDF and TXT documents from the documents/project4 directory, 
then splits them into manageable chunks, 
and then creates vector embeddings using a Hugging Face model. 
These embeddings are stored in a local Chroma database for efficient retrieval.

Finally, the module sets up a Hugging Face language model and a question-answering chain.
When a user submits a query, the system retrieves the most relevant document chunks from the Chroma database,
and uses the language model to generate a response based on the retrieved context.  
"""

import glob
import os
import sys
import time

from dotenv import load_dotenv                                          # Imports load_dotenv from dotenv library to load environment variables from a .env file.
load_dotenv()  # Load environment variables from .env file

from langchain.chains import ConversationalRetrievalChain               # Imports ConversationalRetrievalChain from langchain.chains to create a conversational retrieval chain that combines a language model with a retriever for question-answering tasks.
from langchain.text_splitter import RecursiveCharacterTextSplitter      # Imports RecursiveCharacterTextSplitter from langchain.text_splitter to split text into smaller chunks based on character count and overlap.
from langchain_chroma import Chroma                                     # Imports Chroma from langchain_chroma for creating and managing a vector database for document embeddings.
from langchain_community.document_loaders import ( 
    DirectoryLoader, 
    PyPDFLoader,
    TextLoader
)                                                                       # Imports functions from LangChain for document loading, text splitting, and conversational retrieval chains.
from langchain_core.prompts import ChatPromptTemplate                   # Imports ChatPromptTemplate from langchain_core.promptsto create prompt templates for chat-based interactions with language models.
from langchain_huggingface import HuggingFaceEmbeddings                 # Imports HuggingFaceEmbeddings class from the langchain_huggingface library, which is used to create embeddings for text using Hugging Face models.
from langchain_openai import ChatOpenAI                                 # Imports ChatOpenAI from langchain_openai library to interact with OpenAI's chat-based language models.

# Constants
HF_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")                          # Retrieves Hugging Face API token from environment variables, which is required for authenticating requests to Hugging Face's API.
HF_MODEL = "openai/gpt-oss-20b:fastest"                                 # Specifies Hugging Face model to be used for generating responses. In this case, it is set to "openai/gpt-oss-20b:fastest", which is a specific version of the GPT model optimized for speed.
HF_URL = "https://router.huggingface.co/v1"                             # Specifies base URL for the Hugging Face API endpoint, which is used to send requests to the Hugging Face model for generating responses.
CHROMA_PATH = "vectorstore"                                             # Specifies directory path where the Chroma vector database will be stored. This database is used to store embeddings of the processed documents for efficient retrieval during question-answering tasks.
DOCUMENTS_PATH = "documents_project4"                                   # Specifies directory path where the PDF and TXT documents to be processed are located. The system will load documents from this directory for embedding and retrieval.           

def create_llm(streaming=False):                                        # initialize Hugging Face language model with specified model, API token, and base URL. The streaming parameter determines whether to enable streaming responses from the model.                                        
                                                                       
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is not set. Set your Hugging Face token "
            "in the environment before running the program."
        )
    
    return ChatOpenAI(
        model=HF_MODEL,
        api_key=HF_TOKEN,
        base_url=HF_URL,
        streaming=streaming,
    )

def create_embeddings():                                                # initialize Hugging Face embeddings model with specified model and device. The embeddings are used to convert text into vector representations for efficient retrieval and similarity search.
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
     encode_kwargs={"normalize_embeddings": True},
    )

def get_text_splitter():                                                # Create a RecursiveCharacterTextSplitter instance with specified chunk size, overlap, and length function. This splitter is used to divide documents into smaller chunks for embedding and retrieval.
    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        length_function=len,
        add_start_index=True,
    )

def load_documents():                                                   # Load both PDF and TXT documents from the documents directory, can add additional document types in the future.
    documents = []

    # Load PDF files
    pdf_loader = DirectoryLoader(
        DOCUMENTS_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    pdf_documents = pdf_loader.load()
    documents.extend(pdf_documents)

    # Load TXT files
    txt_loader = DirectoryLoader(
        DOCUMENTS_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    txt_documents = txt_loader.load()
    documents.extend(txt_documents)

    print(f"Loaded {len(pdf_documents)} PDF files")
    print(f"Loaded {len(txt_documents)} TXT files")
    print(f"Total loaded documents: {len(documents)}")

    return documents

def filter_metadata(doc):                                                # Filter out unwanted sections from documents based on metadata. This function checks the 'section' metadata of a document and filters out documents that belong to certain sections (e.g., references, acknowledgments, appendix) that are not relevant for the question-answering task.
    skip_sections = {"references", "acknowledgments", "appendix"}
    section = doc.metadata.get("section", "").lower()
    return not any(s in section for s in skip_sections)

def process_documents(documents, text_splitter):                        # Process and filter documents into chunks. This function takes a list of documents and a text splitter as input, splits the documents into smaller chunks using the text splitter, and filters out unwanted sections based on metadata. It returns a list of filtered document chunks that are ready for embedding and retrieval in the question-answering system.                            
    chunks = text_splitter.split_documents(documents)
    return [chunk for chunk in chunks if filter_metadata(chunk)]

def load_or_create_vectorstore(embeddings):                             # Load or create a Chroma vectorstore for document embeddings. This function checks if a Chroma vectorstore already exists in the specified directory. If it does, it loads the existing vectorstore and updates it with any new documents. If it doesn't exist, it creates a new vectorstore from the documents in the specified directory. The embeddings parameter is used to create vector representations of the document chunks for efficient retrieval during question-answering tasks.
    if os.path.exists(CHROMA_PATH):
        return handle_existing_vectorstore(embeddings)
    return create_new_vectorstore(embeddings)

def handle_existing_vectorstore(embeddings):                            # Handle loading and updating an existing Chroma vectorstore. This function loads an existing Chroma vectorstore from the specified directory, retrieves the documents currently stored in the vectorstore, and checks for any new documents in the documents directory that are not already in the vectorstore. If new documents are found, it processes and adds them to the vectorstore. If no new documents are found, it simply returns the existing vectorstore. The embeddings parameter is used to create vector representations of any new document chunks for efficient retrieval during question-answering tasks.
    """Handle loading and updating an existing vectorstore."""

    print("Loading existing Chroma database...")

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Get documents currently stored in Chroma
    collection = vectorstore.get()

    processed_files = {
        meta.get("source")
        for meta in collection["metadatas"]
        if meta and meta.get("source")
    }

    # Load all PDFs and TXT files from the documents directory
    documents = load_documents()

    if not documents:
        print(f"No documents found in '{DOCUMENTS_PATH}' directory.")
        sys.exit(1)

    # Determine which source files are already in Chroma
    current_files = {
        doc.metadata.get("source")
        for doc in documents
        if doc.metadata.get("source")
    }

    new_files = current_files - processed_files

    if new_files:
        new_documents = [
            doc for doc in documents
            if doc.metadata.get("source") in new_files
        ]

        print(f"Found {len(new_files)} new document files to process...")

        update_vectorstore(
            vectorstore,
            new_documents,
            processed_files
        )
    else:
        print("No new document files to process.")

    return vectorstore

def update_vectorstore(vectorstore, new_documents, processed_files):        # Update an existing Chroma vectorstore with new documents. This function takes an existing Chroma vectorstore, a list of new documents, and a set of already processed files as input. It processes new documents into chunks using the text splitter, filters out unwanted sections based on metadata, and adds the resulting chunks to the vectorstore. If no new documents are found, it simply returns without making any changes. The processed_files parameter is used to keep track of which documents have already been added to the vectorstore to avoid duplicates.

    print(f"Found {len(new_documents)} new document files to process...")

    documents = load_documents()

    new_documents = [
        doc for doc in documents
        if doc.metadata.get("source") not in processed_files
    ]

    if not new_documents:
        print("No new documents to add.")
        return

    print("\nNew documents being added:")

    for doc in new_documents:
        print(f"  - {doc.metadata.get('source')}")

    filtered_chunks = process_documents(
        new_documents,
        get_text_splitter()
    )

    if filtered_chunks:
        print(f"Adding {len(filtered_chunks)} chunks to Chroma...")
        vectorstore.add_documents(filtered_chunks)
        print("Database updated successfully!")


def create_new_vectorstore(embeddings):                                     # Create a new Chroma vectorstore from PDF and TXT documents. This function loads documents from the specified directory, processes them into chunks using the text splitter, and creates a new Chroma vectorstore with the resulting chunks. The embeddings parameter is used to create vector representations of the document chunks for efficient retrieval during question-answering tasks. If no documents are found in the specified directory, it prompts the user to add documents and exits the program.
    print("Creating new Chroma database...")

    documents = load_documents()

    if not documents:
        print(f"No files found in '{DOCUMENTS_PATH}' directory!")
        print(f"Please add your PDF or TXT files to the '{DOCUMENTS_PATH}' directory and run again.")
        sys.exit(1)

    print(f"Loaded {len(documents)} source documents/pages.")
    print("(This may take a while as documents need to be processed and embedded)")

    filtered_chunks = process_documents(
        documents,
        get_text_splitter()
    )

    print(f"Created {len(filtered_chunks)} text chunks.")

    os.makedirs(CHROMA_PATH, exist_ok=True)

    return Chroma.from_documents(
        documents=filtered_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )


def create_qa_chain(llm, vectorstore):                                          # Create a question-answering chain using the language model and vectorstore. This function sets up a prompt template for the question-answering task, which instructs the language model to answer questions based on the provided context from the project documents. It then creates a ConversationalRetrievalChain that combines the language model with the vectorstore retriever, allowing for retrieval-augmented generation (RAG) of answers based on relevant document chunks. The chain is configured to return source documents along with the generated answers for transparency and reference.
    prompt_template = """I am a knowledge-grounded assistant.

    I will answer the question using the provided context from the project documents.

    Rules:
    - I will use the provided context as your primary source.
    - If the answer cannot be found in the context, I will say that the information is not available in the provided documents.
    - I will not invent or assume facts that are not supported by the context.
    - I will be concise and clear.

    Context:
    {context}

    Question:
    {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        chain_type="stuff",
        verbose=True,
    )

def get_rag_response(qa_chain, query, chat_history):        # Get response from RAG-powered chain with streaming. This function takes a question-answering chain, a user query, and the conversation history as input. It invokes the QA chain to retrieve relevant document chunks and generate an answer based on the provided context. The answer is printed character-by-character to simulate streaming, and the sources of the retrieved documents are also displayed for transparency. If any errors occur during the process (e.g., keyboard interrupt or connection error), they are caught and displayed to the user.
    """Gets a response from the RAG chain."""

    print("\n=== RAG Response ===")

    result = qa_chain.invoke({
        "question": query,
        "chat_history": chat_history
    })

    answer = result["answer"]

    print(answer)

    return answer

def main():
    """Main execution function."""
    llm = create_llm(streaming=True)                                                # Initializes Hugging Face language model with streaming enabled.
    embeddings = create_embeddings()                                                # Initializes embedding model.
    vectorstore = load_or_create_vectorstore(embeddings)                            # Loads an existing vectorstore or creates a new one based on the documents in the specified directory.
    qa_chain = create_qa_chain(llm, vectorstore)                                    # Creates the question-answering chain using the language model and vectorstore for retrieval-augmented generation (RAG).      

    chat_history = []                                                               # Initializes an empty list to keep track of the conversation history.

    print("Hi, I'm your AI assistant!")                                             # Prints message to the user.
    #print("Ask a question, or enter 'exit' to end the conversation.\n")             # Prompts user to input a query or message for the assistant.

    while True:
        query = input("Ask a question, or enter 'exit' to end the conversation: ")  # Prompts user to input a query or message for the assistant.

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = get_rag_response(qa_chain, query, chat_history)

        # Add this conversation turn to the chat history
        chat_history.append((query, response))

        print()

if __name__ == "__main__":
    main()