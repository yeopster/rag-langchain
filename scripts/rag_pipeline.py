from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def load_rag_pipeline(
    vector_store_path="data/vector_store",
    embedding_model="all-MiniLM-L6-v2",
    llm_model="distilgpt2"  # Use a lightweight model for testing
):
    # Load embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    model = AutoModelForCausalLM.from_pretrained(llm_model)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        device=0 if torch.cuda.is_available() else -1
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Set up RAG pipeline
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 chunks
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def query_rag(qa_chain, question):
    result = qa_chain({"query": question})
    return result["result"], result["source_documents"]

if __name__ == "__main__":
    qa_chain = load_rag_pipeline()
    question = "How do I configure the firewall in the router manual?"
    answer, sources = query_rag(qa_chain, question)
    print("Answer:", answer)
    print("\nSources:")
    for doc in sources:
        print(f"- {doc.metadata['source']}: {doc.page_content[:100]}...")