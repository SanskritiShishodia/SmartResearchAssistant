import streamlit as st

# 1. RENDER UI FIRST (So you don't see a blank screen while imports load)
st.set_page_config(page_title="Free RAG Assistant")
st.title("🦙 Free Local RAG Assistant")
st.write("Loading AI libraries... please wait...")

# 2. HEAVY IMPORTS (Now happen after the title appears)
import tempfile
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.memory import ConversationBufferMemory
from langchain_classic.chains.conversation.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# Load environment variables from .env file
load_dotenv()

# 3. UPDATE UI
st.success("Libraries loaded!")

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Files")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

# --- Functions ---
def get_text_chunks(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        documents.extend(docs)
        os.remove(tmp_file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def get_vectorstore(text_chunks):
    # Uses CPU-friendly embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # Connects to the Groq server running in the background
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

# --- Main Logic ---
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Process Docs"):
    if not uploaded_files:
        st.error("⚠️ Please upload PDF documents.")
    else:
        with st.spinner("Processing..."):
            chunks = get_text_chunks(uploaded_files)
            vectorstore = get_vectorstore(chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("Ready!")

# --- Chat UI ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask a question:")

if user_question:
    if st.session_state.conversation:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Thinking..."):
            response = st.session_state.conversation({'question': user_question})
            answer = response['answer']
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
                
