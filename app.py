from tempfile import NamedTemporaryFile
import os

import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Resume Helper",
    page_icon="👩🏻‍💻👩🏻",
    layout="wide",  # Changed layout to wide for better visualization
    initial_sidebar_state="auto",
    menu_items=None,
)

# Define sidebar content layout
uploaded_file = st.sidebar.file_uploader("Upload a resume / cover letter")

if uploaded_file:
    bytes_data = uploaded_file.read()
    with NamedTemporaryFile(delete=False) as tmp:  
        tmp.write(bytes_data)  
        reader = PDFReader()
        docs = reader.load_data(tmp.name)
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            model="gpt-3.5-turbo",
            temperature=0.0,
            system_prompt="You are a professional human resource manager in tech. You are taught that a strong resume should be brief but powerful, have strong action verbs (facilitated is not a strong action verb), and show impact of the experience. You will be provided with the document of the candidate's resume or cover letter. Give detailed helpful feedback on what changes can be made to make the resume more effective. You should always tailor your response to the role the candidate is applying for. You should also provide a score out of 100. The score should reflect the quality of the resume for the role. Refer to the candidate as 'you' and the resume as 'your resume'.",
        )
        index = VectorStoreIndex.from_documents(docs)
    os.remove(tmp.name)

    if "chat_engine" not in st.session_state.keys():  
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="condense_question", verbose=False, llm=llm
        )

    st.session_state.file_processed = True

    if "initial_analysis_triggered" not in st.session_state:
        st.session_state.initial_analysis_triggered = True  
        initial_prompt = "Based on my document, how can I make it more effective?"
        st.session_state.messages = [{"role": "user", "content": initial_prompt}]
        initial_response = st.session_state.chat_engine.stream_chat(initial_prompt)
        if initial_response and initial_response.response:
            st.session_state.messages.append({"role": "assistant", "content": initial_response.response})

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi!👋🏻 Need help with your resume? Upload your document to get started!"}]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Generate and display response for the new prompt
    response = st.session_state.chat_engine.stream_chat(prompt)
    if response and response.response:
        st.session_state.messages.append({"role": "assistant", "content": response.response})


for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # Use expanders to organize chat messages
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response.response_gen)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history