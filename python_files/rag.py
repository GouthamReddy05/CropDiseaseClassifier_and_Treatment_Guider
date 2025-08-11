# import os
# os.environ["USE_TF"] = "0"   # Disable TensorFlow imports
# os.environ["USE_TORCH"] = "1"  # Force PyTorch


# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama 
from sentence_transformers import SentenceTransformer
import faiss 
# from python_files.main import process_image, predict_disease
import streamlit as st
import json
import os
from dotenv import load_dotenv
import getpass
from langchain.chat_models import init_chat_model


load_dotenv()


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_vector_db():
    try:
        return faiss.read_index("information.index")
    except Exception as e:
        st.error(f"Failed to load vector database: {e}")
        return None

# model = init_chat_model(
#     "gemini-2.5-flash",
#     model_provider="google_genai",
#     api_key=gemini_api_key
# )
# model.invoke("Hello, world!")

# model = init_chat_model("mistral-large-latest", model_provider="mistralai")


# print(model.invoke("Hello, are you working?"))


def search_faiss(answer_query):
    vector_db = get_vector_db()
    emb = embedding_model.encode(answer_query)
    emb = emb.reshape(1, -1).astype("float32")

    ## search in vector database

    answer = None

    _, indices = vector_db.search(emb, k=1)
    index = indices[0][0]

    with open("structured_data.json", "r") as f:
        metadata = json.load(f)

    if index != -1 and index < len(metadata):
        return metadata[index]
    
    return "Sorry, no relevant information found."




def run_rag_pipeline(answer_query):

    context = search_faiss(answer_query)
    

    return context

    

