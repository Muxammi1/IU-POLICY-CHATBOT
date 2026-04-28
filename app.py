import os
import warnings
import logging
import base64

import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# Load .env for API key

# LangChain & PDF tools
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Disable warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Page Configuration ---
st.set_page_config(page_title="IQRA UNIVERSITY CHATBOT", layout="wide")

# --- Function to Encode Local Image for HTML ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# --- LOAD LOGO ---
# ASSUMPTION: The file is named 'LOGO-IU.png'. 
# If your file is .jpg, change this line to "LOGO-IU.jpg"
logo_file = "LOGO-IU.png" 
logo_base64 = get_base64_of_bin_file(logo_file)

# Logo HTML generation
logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="nav-logo">' if logo_base64 else '<div class="nav-logo-text">IQRA UNIVERSITY</div>'

# --- THEME & ANIMATION CSS ---
st.markdown(f"""
    <style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Global Reset */
    body {{
        background-color: #F8F9FA;
        font-family: 'Poppins', sans-serif;
        margin: 0;
        padding: 0;
    }}

    /* Hide Default Streamlit Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Adjust main container to push content below the fixed navbar */
    .block-container {{
        padding-top: 100px !important; /* Push content down so it doesn't hide behind nav */
        padding-bottom: 5rem;
    }}

    /* --- OFFICIAL NAVBAR STYLING (FULL WIDTH FIX) --- */
    .navbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #002D72; /* IU Navy Blue */
        padding: 15px 40px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-bottom: 5px solid #FFC72C; /* IU Yellow */
        
        /* FIXED POSITIONING TO SPAN FULL WIDTH */
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        box-sizing: border-box; /* Ensures padding doesn't break width */
        z-index: 99999;
        
        animation: slideDown 0.8s ease-out;
    }}

    .nav-left {{
        display: flex;
        align-items: center;
    }}

    .nav-logo {{
        height: 50px; 
        margin-right: 20px;
        transition: transform 0.3s ease;
    }}
    
    .nav-logo:hover {{
        transform: scale(1.05);
    }}

    .nav-title {{
        color: white;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}

    .nav-links {{
        display: flex;
        gap: 20px;
    }}

    .nav-link {{
        color: rgba(255,255,255,0.8);
        text-decoration: none;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: color 0.3s;
    }}

    .nav-link:hover {{
        color: #FFC72C;
    }}

    /* --- ANIMATIONS --- */
    @keyframes slideDown {{
        from {{ transform: translateY(-100%); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}

    @keyframes fadeInUp {{
        from {{ transform: translateY(20px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}

    /* --- CHAT AREA --- */
    .stChatMessage {{
        background-color: white !important;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #EAEAEA;
        animation: fadeInUp 0.5s ease-out;
        margin-bottom: 10px;
    }}

    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {{
        border-left: 5px solid #FFC72C; /* Yellow accent for user */
    }}
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {{
        border-left: 5px solid #002D72; /* Blue accent for bot */
    }}

    /* Input Field Styling */
    .stChatInputContainer {{
        padding-bottom: 30px;
    }}
    
    div[data-testid="stChatInput"] {{
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        border-radius: 20px;
    }}

    /* Hero Text in Body */
    .hero-text {{
        text-align: center;
        margin-top: 20px;
        margin-bottom: 40px;
        animation: fadeInUp 1s ease-out;
    }}
    
    .hero-title {{
        color: #002D72;
        font-size: 32px;
        font-weight: 800;
    }}
    
    .hero-subtitle {{
        color: #666;
        font-size: 16px;
    }}

    /* --- MOBILE RESPONSIVENESS --- */
    @media (max-width: 600px) {{
        .navbar {{ padding: 10px 20px; }}
        .nav-title {{ font-size: 18px; }}
        .nav-logo {{ height: 35px; }}
        .nav-links {{ display: none; }}
        .block-container {{ padding-top: 80px !important; }}
    }}
    </style>

    <div class="navbar">
        <div class="nav-left">
            {logo_html}
        </div>
        <div class="nav-links">
            <span class="nav-link">PORTAL</span>
            <span class="nav-link">ADMISSIONS</span>
            <span class="nav-link">ACADEMICS</span>
            <span class="nav-link">CONTACT</span>
        </div>
    </div>
    
    <div class="hero-text">
        <div class="hero-title">How can we help you today?</div>
        <div class="hero-subtitle">Ask about admissions, policies, or campus life.</div>
    </div>
""", unsafe_allow_html=True)

# --- Chat History Setup ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages with container styling
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

# --- Load PDFs & Vector Store ---
@st.cache_resource
def get_vectorstore():
    pdf_folder = "./policies"
    loaders = []
    
    if os.path.exists(pdf_folder):
        for filename in os.listdir(pdf_folder):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(pdf_folder, filename)
                loaders.append(PyPDFLoader(file_path))
    else:
        os.makedirs(pdf_folder)
        return None

    if not loaders:
        return None

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)

    return index.vectorstore

# --- Chat Input ---
prompt = st.chat_input("Type your question here...")

if prompt:
    # Display user message immediately
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Logic
    groq_sys_prompt = ChatPromptTemplate.from_template("""
        You are an official assistant for Iqra University. 
        You are professional, accurate, and helpful.
        Answer the following Question: {user_prompt}.
        Start the answer directly.
    """)

    model = "llama-3.1-8b-instant"

    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model
    )

    try:
        vectorstore = get_vectorstore()
        
        if vectorstore is None:
            # Fallback if no PDFs
            response = "I currently don't have the policy documents loaded. Please upload PDFs to the 'policies' folder."
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
        else:
            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type='stuff',
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True
            )

            result = chain({"query": prompt})
            response = result["result"]

            # Display assistant response
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"Error: {str(e)}")