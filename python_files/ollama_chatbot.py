import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add project root


from python_files.rag import run_rag_pipeline
import python_files.main as main
from PIL import Image
import os

st.set_page_config(page_title="Ollama Chatbot", page_icon="🤖")
st.title("🤖 Ollama LLM Chatbot (with LangChain + Memory)")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "llm_chain" not in st.session_state:
    llm = ChatOllama(model="llama3")
    st.session_state.llm_chain = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False
    )

if st.session_state.memory.chat_memory.messages:
    st.markdown("### 💬 Chat History")
    for msg in st.session_state.memory.chat_memory.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

input_placeholder = st.empty()
with input_placeholder.container():
    col_input, col_plus, col_send = st.columns([10, 3, 2])
    with col_input:
        user_input = st.text_input("", placeholder="Type your message or question", label_visibility="collapsed")
    with col_plus:
        show_uploader = st.button("Add Image", use_container_width=True)
    with col_send:
        send_clicked = st.button("Send", use_container_width=True)

    uploaded_file = None
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False
    if show_uploader:
        st.session_state.show_uploader = not st.session_state.show_uploader
    if st.session_state.show_uploader:
        uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if send_clicked:
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img.save("temp_leaf.jpg")
        predicted_disease = main.predict_disease("temp_leaf.jpg")
        st.markdown(f"**Predicted Disease:** {predicted_disease}")

        rag_output = run_rag_pipeline(predicted_disease)
        with st.chat_message("assistant"):
            st.markdown(rag_output)

if st.button("🗑️ Clear Chat"):
    st.session_state.memory.clear()
    if os.path.exists("temp_leaf.jpg"):
        os.remove("temp_leaf.jpg")
    st.session_state.show_uploader = False
    st.rerun()
