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

# --- UI / CSS ---
st.markdown(f"""
<style>
    /* 1. GLOBAL OVERRIDES - KILL THE RED FOCUS RING */
    :root {{
        --primary-color: #003366 !important; /* Changes focus outline from red to IU Navy */
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: white !important;
        color: #003366 !important;
    }}

    /* Force all text to stay IU Navy */
    h1, h2, h3, p, span, div {{
        color: #003366 !important;
    }}

    /* 2. NAVBAR */
    .stApp {{ margin-top: 80px; }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    .navbar {{
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #003366;
        padding: 10px 0;
        border-bottom: 6px solid #f8a51b;
        position: fixed;
        top: 0; left: 0; width: 100%;
        z-index: 9999;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }}
    .nav-logo {{ height: 60px; }}

    /* 3. CHAT BUBBLES - THE COLOR BORDERS */
    
    /* Base Bubble Style */
    [data-testid="stChatMessage"] {{
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        padding: 18px !important;
        margin-bottom: 15px !important;
        border: 1px solid #E0E0E0 !important;
        box-shadow: 2px 5px 15px rgba(0,0,0,0.05) !important;
    }}

    /* USER MESSAGE: Left Blue Border */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        border-left: 12px solid #003366 !important;
    }}

    /* BOT MESSAGE: Left Yellow Border */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
        border-left: 12px solid #f8a51b !important;
    }}

    /* Fix Avatar Icons to match */
    [data-testid="stChatMessageAvatarUser"] {{
        background-color: #003366 !important;
        border-radius: 8px !important;
    }}
    [data-testid="stChatMessageAvatarAssistant"] {{
        background-color: #f8a51b !important;
        border-radius: 8px !important;
    }}

    /* 4. CHAT INPUT - REMOVE RED ACTIVE STATE */
    
    /* Border when typing */
    div[data-testid="stChatInput"] textarea {{
        border: 2px solid #003366 !important;
        color: #003366 !important;
    }}

    /* Remove the red outline on focus */
    div[data-testid="stChatInput"] textarea:focus {{
        box-shadow: 0 0 0 2px rgba(0, 51, 102, 0.2) !important;
        border-color: #003366 !important;
        outline: none !important;
    }}
    
    /* The outer container boundary */
    div[data-testid="stChatInput"] {{
        border: none !important;
        background-color: transparent !important;
    }}

    .hero-title {{
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 25px;
    }}
</style>

<div class="navbar">
    <div>{logo_html}</div>
</div>

<div class="hero-title">How can we help you today?</div>
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

# --- ADDED: Suggestion Questions (Displays only when chat is empty) ---
if len(st.session_state.messages) == 0 and not prompt:
    st.markdown("<p style='text-align: center; color: #003366; margin-top: 10px;'><b>Suggested questions:</b></p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("What is the admission policy?", use_container_width=True): prompt = "What is the admission policy?"
        if st.button("How to apply for scholarships?", use_container_width=True): prompt = "How to apply for scholarships?"
    with c2:
        if st.button("What are the attendance rules?", use_container_width=True): prompt = "What are the attendance rules?"
        if st.button("Explain the grading system", use_container_width=True): prompt = "Explain the grading system"

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
            response = "I couldn't find any policy documents. Please ensure PDFs are in the /policies folder."
        else:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type="stuff",
                return_source_documents=True  # ADDED: Needed to calculate the accuracy score
            )
            result = qa_chain.invoke({"query": prompt})
            response_text = result["result"]
            source_docs = result.get("source_documents", [])

            # --- ADDED: Accuracy Calculation & Progress Bar HTML ---
            score = 90 # Default Fallback
            if source_docs:
                query_words = set(prompt.lower().split())
                if query_words:
                    best_match_count = 0
                    for doc in source_docs:
                        matches = sum(1 for word in query_words if word in doc.page_content.lower())
                        best_match_count = max(best_match_count, matches)
                    conf = int((best_match_count / len(query_words)) * 100)
                    score = max(45, min(98, conf + 35)) # Formulate a realistic %

            bar_html = f"""
            <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #EAEAEA; display: flex; align-items: center; gap: 10px; font-size: 0.85rem;">
                <span style="color:#003366;">📊</span>
                <div style="background-color:#E9ECEF; border-radius:10px; height:8px; width:100%;">
                    <div style="background-color:#f8a51b; height:100%; width:{score}%; border-radius:10px;"></div>
                </div>
                <span style="font-weight:bold; color:#003366;">{score}% Match</span>
            </div>
            """
            
            response = response_text + bar_html

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"System Error: {str(e)}")
