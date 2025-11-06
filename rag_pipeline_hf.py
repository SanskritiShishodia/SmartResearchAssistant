from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def load_and_chunk_docs(pdf_paths):
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    return chunks

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # free embedding model
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

def get_rag_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # Free local LLM
    hf_pipeline = pipeline(
        "text-generation",
        model="google/flan-t5-small",  # free HF model
        max_length=512,
        do_sample=True,
        temperature=0.7
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
