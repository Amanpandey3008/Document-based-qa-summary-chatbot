import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from openai._client import OpenAI
import tempfile
import os

# Initialize OpenAI and Pinecone clients
openclient = OpenAI(api_key='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')
pc = Pinecone(api_key='39560f3b-a5a2-461d-8160-2dc01fb24044')
index = pc.Index('serverless-index')


# Function to process PDF and store chunks
def process_and_store_pdf(uploaded_file):
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Save uploaded PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())

            # Load PDF content from the temporary file
            docs = chunk(tmp_file.name)

            # Clean up the temporary file
            os.unlink(tmp_file.name)

            if docs:
                storing(docs)
                st.success("PDF processing complete!")
            else:
                st.error("Failed to process PDF. Please try again.")
    else:
        st.error("Please upload a PDF file.")


# Function to chunk text
def chunk(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    return docs


# Function to store embeddings in Pinecone index
def storing(docs):
    chunks = []
    for i in docs:
        chunks.append(i.page_content)
    text = " ".join(chunks)
    if len(text) > 20000:
        chunk_size = len(text) // 20
    else:
        chunk_size = len(text) // 10
    db_chunks = []
    for i in range(10):
        start = i * chunk_size
        end = start + chunk_size
        if end < len(text):
            chunk = text[start:end]
            db_chunks.append(chunk)
        else:
            chunk = text[start:]
            db_chunks.append(chunk)
    temp = []
    for i, j in enumerate(db_chunks):
        response = openclient.embeddings.create(input=j, model="text-embedding-ada-002")
        temp_dict = {}
        temp_dict["id"] = str(i + 1)
        temp_dict["values"] = response.data[0].embedding
        temp_dict["metadata"] = {"Chunk": j}
        temp.append(temp_dict)
    index.upsert(vectors=temp)


# Streamlit app layout and logic
def main():
    st.title("Text Processing and QA System")

    # Sidebar for uploading PDF
    uploaded_file = st.sidebar.file_uploader("Upload PDF file", type="pdf")

    # Main section
    if st.button("Process PDF"):
        process_and_store_pdf(uploaded_file)


if __name__ == "__main__":
    main()
