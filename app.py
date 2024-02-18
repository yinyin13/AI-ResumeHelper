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
    page_icon=":female-technologist::skin-tone-2:",
    layout="centered",
    menu_items=None,
)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello!👋🏻 Need help with your resume? Upload your document to get started!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

uploaded_file = st.file_uploader("Upload your resume")
if uploaded_file:
    bytes_data = uploaded_file.read()
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(bytes_data)
        with st.spinner(text="Loading and indexing the resume – hang tight! This should take 1-2 minutes."):
            reader = PDFReader()
            docs = reader.load_data(tmp.name)
            st.session_state['llm'] = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
                model="gpt-3.5-turbo",
                temperature=0.0,
                system_prompt="You are a professional human resource manager in tech. You are taught that a strong resume should be brief but powerful, have strong action verbs (facilitated is not a strong action verb), and show impact of the experience. You will be provided with the document of the candidate's resume. Give detailed helpful feedback on what changes can be made to make the resume more effective. You should always tailor your response to the role the candidate is applying for. You should also provide a score out of 100. The score should reflect the quality of the resume for the role. Refer to the candidate as 'you' and the resume as 'your resume'.",
            )
            index = VectorStoreIndex.from_documents(docs)
    os.remove(tmp.name)

    if "chat_engine" not in st.session_state.keys():
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=False, llm=st.session_state['llm'])

    with st.chat_message("assistant"):
        with st.spinner("Reading and evaluating the resume..."):
            response = st.session_state.chat_engine.stream_chat("Please evaluate the resume and mention what role will this resume be good for.")
            st.write_stream(response.response_gen)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
            st.session_state['llm'].system_prompt = "You are a professional human resource manager in tech who can provide answers to the user's questions based on their resume."

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.spinner("Thinking..."):
        response = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response.response_gen)
        message = {"role": "assistant", "content": response.response}
        st.session_state.messages.append(message)
        with st.chat_message("assistant"):
            st.write(response.response)