import os
import warnings
import logging
import base64
import streamlit as st

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- API KEY ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- Disable warnings ---
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Page Config ---
st.set_page_config(page_title="IU Chatbot | Support", layout="wide")

# --- Image Encoder ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

# --- Logo & IU Branding Colors ---
# Navy: #003366 | Gold: #f8a51b
logo_file = "LOGO-IU.png"
logo_base64 = get_base64_of_bin_file(logo_file)
logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="nav-logo">' if logo_base64 else '<div class="nav-logo-text">IQRA UNIVERSITY</div>'

# --- UI / CSS OVERHAUL ---
st.markdown(f"""
<style>
    /* 1. FORCE LIGHT MODE & PREVENT DARK MODE OVERRIDE */
    :root {{
        --primary-color: #003366;
        --background-color: #FFFFFF;
        --secondary-background-color: #F0F2F6;
        --text-color: #003366;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: white !important;
        color: #003366 !important;
    }}

    /* Target all headers and text to stay dark blue */
    h1, h2, h3, p, span, div {{
        color: #003366 !important;
    }}

    /* 2. NAVBAR STYLING */
    .stApp {{
        margin-top: 80px;
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .navbar {{
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #003366;
        padding: 10px 0;
        border-bottom: 6px solid #f8a51b;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 9999;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }}

    .nav-logo {{
        height: 60px;
    }}

    .nav-logo-text {{
        color: white !important;
        font-weight: bold;
        font-size: 24px;
        letter-spacing: 1px;
    }}

    /* 3. HERO SECTION */
    .hero-container {{
        text-align: center;
        padding: 40px 20px 20px 20px;
    }}

    .hero-title {{
        font-size: 32px;
        font-weight: 800;
        color: #003366 !important;
        margin-bottom: 10px;
    }}

    .hero-subtitle {{
        font-size: 16px;
        color: #555 !important;
    }}

    /* 4. CHAT INPUT STYLING */
    .stChatInputContainer {{
        padding-bottom: 20px !important;
        background-color: transparent !important;
    }}
    
    div[data-testid="stChatInput"] {{
        border: 2px solid #003366 !important;
        border-radius: 10px !important;
    }}

    /* 5. CHAT MESSAGE BUBBLES */
    [data-testid="stChatMessage"] {{
        background-color: #f8f9fa !important;
        border-radius: 15px !important;
        padding: 15px !important;
        margin-bottom: 10px !important;
        border: 1px solid #e0e0e0 !important;
    }}
</style>

<div class="navbar">
    <div>{logo_html}</div>
</div>

<div class="hero-container">
    <div class="hero-title">How can we help you today?</div>
    <div class="hero-subtitle">Ask about IU policies, admissions, or campus details.</div>
</div>
""", unsafe_allow_html=True)

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f'<div style="color: #003366;">{msg["content"]}</div>', unsafe_allow_html=True)

# --- VECTOR STORE (RAG Pipeline) ---
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# --- Chat Input Logic ---
prompt = st.chat_input("Type your question here...")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    model = "llama-3.1-8b-instant"
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model)

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            response = "I couldn't find any policy documents. Please ensure PDFs are in the `/policies` folder."
        else:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type="stuff"
            )
            result = qa_chain.invoke({"query": prompt})
            response = result["result"]

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"System Error: {str(e)}")
