import streamlit as st
import os
from rag_pipeline_hf import load_and_chunk_docs, create_vectorstore  # your local RAG code
from gpt3_module import generate_answer_with_context  # GPT-3.5 wrapper

st.set_page_config(page_title="🧠 Smart Research Assistant (GPT-3.5 + RAG)", layout="wide")
st.title("🧠 Smart Research Assistant (GPT-3.5 + RAG)")

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Upload PDFs
pdfs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
query = st.text_input("Ask a question about the documents:")

if pdfs and query:
    with st.spinner("Processing your question..."):
        # Save uploaded PDFs
        os.makedirs("data", exist_ok=True)
        paths = []
        for pdf in pdfs:
            path = f"data/{pdf.name}"
            with open(path, "wb") as f:
                f.write(pdf.read())
            paths.append(path)

        # Load, chunk, and create vectorstore
        chunks = load_and_chunk_docs(paths)
        vectordb = create_vectorstore(chunks)

        # Retrieve top 3 relevant chunks for RAG
        docs = vectordb.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate answer using GPT-3.5 with retrieved context
        answer = generate_answer_with_context(query, context, max_tokens=200)

        # Append new entry to session history
        st.session_state.history.append({
            "query": query,
            "answer": answer,
            "sources": docs
        })

# Display latest answer
if st.session_state.history:
    latest = st.session_state.history[-1]
    st.markdown("### 🧾 Answer")
    st.write(latest["answer"])

    st.markdown("### 📚 Sources")
    for doc in latest["sources"]:
        with st.expander(doc.metadata.get("source", "PDF Document")):
            st.write(doc.page_content[:500] + "...")  # first 500 chars

# Display full conversation history
if st.session_state.history:
    st.markdown("### 📝 Conversation History")
    for idx, item in enumerate(st.session_state.history):
        st.write(f"**Q{idx+1}: {item['query']}**")
        st.write(f"A{idx+1}: {item['answer']}")
