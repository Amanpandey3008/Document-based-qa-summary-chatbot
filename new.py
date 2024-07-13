import streamlit as st
import pymongo
from pymongo import MongoClient
from passlib.hash import pbkdf2_sha256

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['user_db']
users_collection = db['users']

# Function to check if user exists
def user_exists(email):
    return users_collection.find_one({"email": email}) is not None

# Function to register a new user
def register_user(name, email, password):
    hashed_password = pbkdf2_sha256.hash(password)
    users_collection.insert_one({"name": name, "email": email, "password": hashed_password})

# Function to authenticate user
def authenticate_user(email, password):
    user = users_collection.find_one({"email": email})
    if user and pbkdf2_sha256.verify(password, user["password"]):
        return True
    return False


import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from openai._client import OpenAI

# Initialize OpenAI and Pinecone clients
openclient = OpenAI(api_key='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')
pc = Pinecone(api_key='39560f3b-a5a2-461d-8160-2dc01fb24044')
index = pc.Index('testing')

import openai
client = openai.OpenAI(
    api_key="44a4ccc735c530913f33ee596b487cf680388f2901dcbafa3a16eeb71c14786c",
    base_url="https://api.together.xyz/v1",
    )

# Function to process PDF and store chunks
def process_and_store_pdf(path_docs):
    print(path_docs)
    docs = chunk(path_docs)
    print("chunk created")
    storing(docs)
    print("data stored")

# Function to chunk text
def chunk(path_docs):
    loader = PyPDFLoader(path_docs)
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

# Function to interactively get chapter details
def get_chapter_details():
    chapter_number = st.number_input("Total numbers of chapters present:", min_value=1, step=1)
    chapters = []
    for i in range(chapter_number):
        start = st.number_input(f"Starting page of chapter {i + 1}:", min_value=1, step=1)
        end = st.number_input(f"Ending page of chapter {i + 1}:", min_value=start, step=1)
        chapters.append({"starting_page": start, "end_page": end})
    return chapters

# Function to chunk text per chapter
def chunking_per_chapter(docs, chapters):
    text_chapters = []
    for chapter in chapters:
        chapter_chunk = []
        for doc in docs:
            if chapter["starting_page"] < doc.metadata["page"] < chapter["end_page"]:
                chapter_chunk.append(doc.page_content)
        text_chapters.append("".join(chapter_chunk))
    return text_chapters

# Function to generate summary per chapter
def summary_per_chapter(text_chapters):
    chapter_summaries = []
    for text in text_chapters:
        user_content = f"generate a detailed summary of the following chapter in bullet points: {text}"
        output = client.chat.completions.create(model="mistralai/Mistral-7B-Instruct-v0.2", messages=[{"role": "user", "content": user_content}], temperature=0.4, stop=['[/INST]', '</s>'])
        chapter_summaries.append(output.choices[0].message.content)
    return chapter_summaries

# Function to generate questions per chapter
def question_per_chapter(text_chapters):
    qnas = []
    for text in text_chapters:
        system_content = "you are a question and answer generator bot which creates descriptive questions only according to the given Text. Note: don't create multiple-choice questions"
        user_content = f"according to the provided text, create questions based on what when where why how from the following text and generate their answers as well: {text}"
        output = client.chat.completions.create(model="mistralai/Mistral-7B-Instruct-v0.2", messages=[{"role": "system", "content": system_content}, {"role": "user", "content": user_content}], temperature=0.4, stop=['[/INST]', '</s>'])
        qnas.append(output.choices[0].message.content)
    return qnas

# Function to perform question answering
def rag(query):
    response = openclient.embeddings.create(input=query, model="text-embedding-ada-002")
    res = index.query(vector=response.data[0].embedding, top_k=1, include_metadata=True)["matches"]
    text = res[0]["metadata"]["Chunk"]
    system_content = "you are a question-answer bot which answers the user's query according to the given Text only"
    user_content = f"according to the provided text: {text} answer the following question: {query}"
    output = client.chat.completions.create(model="togethercomputer/Llama-2-13b-chat", messages=[{"role": "system", "content": system_content}, {"role": "user", "content": user_content}], temperature=0.4, max_tokens=500, stop=['[/INST]', '</s>'])
    return output.choices[0].message.content

# Streamlit app layout and logic


# Streamlit app layout and logic
def main():
    st.title("Text Processing and QA System")

    # Authentication logic
    is_new_user = st.sidebar.checkbox("New User?")
    if is_new_user:
        st.subheader("Register")
        name = st.text_input("Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            if not user_exists(email):
                register_user(name, email, password)
                st.success("Registration successful. Please log in.")
            else:
                st.warning("User with this email already exists. Please log in.")
    else:
        st.subheader("Log In")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Log In"):
            if authenticate_user(email, password):
                st.success("Log in successful.")
                file_path = st.text_input("Enter local file path:")
                if st.button("Process PDF"):
                    if file_path:
                        process_and_store_pdf(file_path)
                    else:
                        st.warning("Please enter a valid file path.")
            else:
                st.warning("Authentication failed. Please try again.")

    # Main section
    task = st.sidebar.selectbox("Choose a task", ("Chapter-wise Analysis", "Question Answering"))

    if task == "Chapter-wise Analysis":
        chapters = get_chapter_details()
        if st.button("Generate Summary per Chapter"):
            docs = chunk(file_path) if file_path else None
            if docs:
                text_chapters = chunking_per_chapter(docs, chapters)
                summaries = summary_per_chapter(text_chapters)
                for i, summary in enumerate(summaries):
                    st.write(f"Summary of Chapter {i + 1}: {summary}")

    elif task == "Question Answering":
        chapters = get_chapter_details()
        if st.button("Generate Questions per Chapter"):
            docs = chunk(file_path) if file_path else None
            if docs:
                text_chapters = chunking_per_chapter(docs, chapters)
                questions = question_per_chapter(text_chapters)
                for i, qna in enumerate(questions):
                    st.write(f"Questions for Chapter {i + 1}:\n{qna}")

        query = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if file_path:
                answer = rag(query)
                st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()
