import os
import warnings
import logging
import base64

import streamlit as st

# --- API KEY ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- LangChain (UPDATED IMPORTS) ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- Disable warnings ---
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Page Config ---
st.set_page_config(page_title="IQRA UNIVERSITY CHATBOT", layout="wide")

# --- Image Encoder ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

# --- Logo ---
logo_file = "LOGO-IU.png"
logo_base64 = get_base64_of_bin_file(logo_file)

logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="nav-logo">' if logo_base64 else '<div class="nav-logo-text">IQRA UNIVERSITY</div>'

# --- UI / CSS ---
st.markdown(f"""
<style>
body {{
    background-color: #F8F9FA;
    font-family: 'Poppins', sans-serif;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

.block-container {{
    padding-top: 100px !important;
}}

.navbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background-color: #002D72;
    padding: 15px 40px;
    border-bottom: 5px solid #FFC72C;
    position: fixed;
    top: 0;
    width: 100%;
    z-index: 9999;
}}

.nav-logo {{
    height: 50px;
}}

.hero-text {{
    text-align: center;
    margin: 30px;
}}

.hero-title {{
    font-size: 28px;
    font-weight: bold;
    color: #002D72;
}}
</style>

<div class="navbar">
    <div>{logo_html}</div>
</div>

<div class="hero-text">
    <div class="hero-title">How can we help you today?</div>
</div>
""", unsafe_allow_html=True)

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# --- VECTOR STORE (UPDATED PIPELINE) ---
@st.cache_resource
def get_vectorstore():
    pdf_folder = "./policies"
    documents = []

    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        return None

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L12-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# --- Chat Input ---
prompt = st.chat_input("Ask about policies, admissions, etc...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    model = "llama-3.1-8b-instant"

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model
    )

    try:
        vectorstore = get_vectorstore()

        if vectorstore is None:
            response = "No policy PDFs found. Please upload documents in the /policies folder."
        else:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type="stuff"
            )

            result = qa_chain.invoke({"query": prompt})
            response = result["result"]

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error: {str(e)}")
