import streamlit as st
from scripts.rag_pipeline import load_rag_pipeline, query_rag

st.title("Trading Q&A")
st.write("Ask questions about trading, and get accurate answers!")

# Load RAG pipeline
@st.cache_resource
def get_qa_chain():
    return load_rag_pipeline()

qa_chain = get_qa_chain()

# User input
question = st.text_input("Enter your question:", placeholder="e.g., What is fundamental analysis?")
if question:
    with st.spinner("Generating answer..."):
        answer, sources = query_rag(qa_chain, question)
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Sources")
        for doc in sources:
            st.write(f"- {doc.metadata['source']}: {doc.page_content[:200]}...")