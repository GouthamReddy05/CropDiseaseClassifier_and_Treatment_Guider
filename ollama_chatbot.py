import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from rag import run_rag_pipeline
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import main
import os
from dotenv import load_dotenv

load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

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


# Place input bar at the bottom using st.empty()
input_placeholder = st.empty()
with input_placeholder.container():
    col_input, col_plus, col_send = st.columns([10, 3, 2])
    with col_input:
        user_input = st.text_input("", placeholder="Type your message or question", label_visibility="collapsed")
    with col_plus:
        show_uploader = st.button("add files", use_container_width=True)
    with col_send:
        send_clicked = st.button("Send", use_container_width=True)

    uploaded_file = None
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False
    if show_uploader:
        st.session_state.show_uploader = not st.session_state.show_uploader
    if st.session_state.show_uploader:
        uploaded_file = st.file_uploader("Drag and drop a leaf image here", type=["jpg", "jpeg", "png"], label_visibility="visible")

if send_clicked and (user_input or uploaded_file is not None):
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img.save("temp_leaf.jpg")

        predicted_disease = main.predict_disease("temp_leaf.jpg")
        st.markdown(f"**Predicted Disease:** {predicted_disease}")

        rag_output = run_rag_pipeline(str(predicted_disease))
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st.markdown(rag_output)
    elif not uploaded_file:
        st.warning("Please upload a leaf image to get a prediction and context.")

import os

if st.button("🗑️ Clear Chat"):
    st.session_state.memory.clear()
    # Remove uploaded image file if exists
    if os.path.exists("temp_leaf.jpg"):
        os.remove("temp_leaf.jpg")
    # Reset uploader state
    st.session_state.show_uploader = False
    st.session_state["uploaded_file"] = None
    st.rerun()
