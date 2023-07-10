from flask import Flask, request, jsonify
import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Load documents from a directory
def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

# Split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Load OpenAI embeddings
def load_embeddings():
    client = openai.api_key = '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'
    embeddings = OpenAIEmbeddings(model="ada", client=client)
    return embeddings

# Initialize Pinecone index
def init_index(docs, embeddings):
    pinecone.init(api_key="f3493788-2a36-48ee-a2d3-f6205e2c71c0", environment="asia-northeast1-gcp")
    index_name = "qabot"
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return index

# Load language model
def load_language_model():
    client = openai.api_key = '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'
    llm = OpenAI(model="text-davinci-003", client=client)
    return llm

# Initialize question-answering chain
def initialize_qa_chain(llm):
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

# Load language model
llm = load_language_model()

# Initialize question-answering chain
chain = initialize_qa_chain(llm)

# Get similar documents based on a query
def get_similar_docs(query, k=1, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

# Get the answer to a question
def get_answer(query):
    similar_docs = get_similar_docs(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

# Define the API endpoint for answering questions
@app.route('/answer', methods=['POST'])
def answer_question():
    # Get the query from the request data
    query = request.json['query']
    
    # Call the get_answer function
    answer = get_answer(query)
    
    # Return the answer as a JSON response
    return jsonify({'answer': answer})

if __name__ == '__main__':
    # Set the directory path for documents
    directory = 'data'

    # Load documents from the directory
    documents = load_docs(directory)

    # Split documents into chunks
    docs = split_docs(documents)

    # Load OpenAI embeddings
    embeddings = load_embeddings()

    # Initialize Pinecone index
    index = init_index(docs, embeddings)

    # Load language model
    llm = load_language_model()

    # Load question-answering chain
    chain = load_qa_chain(llm)

    # Run the Flask app
    app.run(debug=True)
