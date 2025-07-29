import streamlit as st
from openai import OpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
import glob

# Load environment variables
load_dotenv()

# NVIDIA API client initialization
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")  # Your NVIDIA API key from .env
)

# Pinecone configuration
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = "financial-data-index"
pinecone_namespace = "financial_docs"

# Initialize Pinecone client
try:
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
except Exception as e:
    st.error(f"Failed to initialize Pinecone client or access the index: {e}")
    st.stop()

# Streamlit UI setup
st.title("Financial Document Chatbot")
st.write("Upload your financial PDFs, embed them into Pinecone, and ask questions interactively!")

# Sidebar for file upload
st.sidebar.title("Document Management")
uploaded_files = st.sidebar.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

# Initialize session states
if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to chunk and upsert documents
def vector_embedding():
    pdf_directory = "./financial_documents"
    os.makedirs(pdf_directory, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(pdf_directory, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
    
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    if not pdf_files:
        st.error("No PDF files found in the directory. Please upload valid PDFs.")
        return

    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    vectors = []
    for i, chunk in enumerate(chunks):
        doc_id = chunk.metadata.get("source", f"doc{i}")
        chunk_id = f"{doc_id}#chunk{i + 1}"
        embedding = NVIDIAEmbeddings().embed_documents([chunk.page_content])[0]
        vectors.append({
            "id": chunk_id,
            "values": embedding,
            "metadata": {
                "source": doc_id,
                "chunk_number": i + 1,
                "content": chunk.page_content
            }
        })

    try:
        index.upsert(vectors=vectors, namespace=pinecone_namespace)
        st.session_state.vector_store_ready = True
        st.success("Pinecone vector store created successfully!")
    except Exception as e:
        st.error(f"Failed to create Pinecone vector store: {e}")

# Function to query NVIDIA LLaMA model
def query_llama(prompt):
    completion = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True
    )

    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
    return response

# Embedding documents button
if uploaded_files and st.sidebar.button("Embed Documents"):
    vector_embedding()

# Chatbot UI
st.header("Chat with Your Financial Data")
if st.session_state.vector_store_ready:
    st.chat_input("Type your question here", key="user_query")

    if "user_query" in st.session_state and st.session_state.user_query:
        user_query = st.session_state.user_query
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Query Pinecone
        try:
            pinecone_response = index.query(
                namespace=pinecone_namespace,
                vector=NVIDIAEmbeddings().embed_documents([user_query])[0],
                top_k=5,
                include_values=True,
                include_metadata=True
            )

            # Combine top chunks and send to LLaMA for summarization
            top_chunks = [match['metadata']['content'] for match in pinecone_response["matches"]]
            if top_chunks:
                prompt = f"""
                You are a financial expert. Use the following document chunks to answer the query concisely.
                Query: {user_query}
                Document Chunks:
                {top_chunks}
                Provide a clear and accurate response.
                """
                assistant_response = query_llama(prompt)
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": "No relevant information found in the documents."})
        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error querying Pinecone: {e}"})

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])
else:
    st.info("Please upload and embed documents to start the chatbot.")
