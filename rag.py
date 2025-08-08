import os
os.environ["USE_TF"] = "0"   # Disable TensorFlow imports
os.environ["USE_TORCH"] = "1"  # Force PyTorch


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama 
from sentence_transformers import SentenceTransformer
import faiss 
from main import process_image, predict_disease
import streamlit as st
import json
from dotenv import load_dotenv


# load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")





def get_vector_db():
    try:
        return faiss.read_index("information.index")
    except Exception as e:
        st.error(f"Failed to load vector database: {e}")
        return None



def search_faiss(answer_query):
    vector_db = get_vector_db()
    emb = embedding_model.encode(answer_query)
    emb = emb.reshape(1, -1).astype("float32")

    ## search in vector database

    answer = None

    _, indices = vector_db.search(emb, k=1)
    index = indices[0][0]

    with open("metadata.json", "r") as f:
        metadata = json.load(f)

    if index != -1 and index < len(metadata):
        return metadata[index]
    
    return "Sorry, no relevant information found."



def create_ollama_llm():
    return Ollama(
        base_url="http://localhost:11434",  # Default Ollama URL
        model="llama3",  # Use the model you pulled
    )

# print(run_rag_pipeline("Potato___Late_blight"))

def run_rag_pipeline(answer_query):

    context = search_faiss(answer_query)

    prompt = ChatPromptTemplate.from_template("""
    You are an agricultural crop disease expert.
    Use the following retrieved information to answer the question.

    Retrieved Information:
    {context}

    Question:
    {question}

    Answer in a clear and detailed manner without referring to any links.
    """)

    llm = create_ollama_llm()

    chain = prompt | llm | StrOutputParser()

    final_answer = chain.invoke({
        "context": context,
        "question": answer_query
    })

    return final_answer


