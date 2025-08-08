from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama 
from sentence_transformers import SentenceTransformer
import faiss 
from main import process_image, predict_disease
import streamlit as st
import os
import json
from dotenv import load_dotenv


load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


def get_vector_db():
    return faiss.read_index("information.index")



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

    llm = Ollama(model="llama3")  

    chain = prompt | llm | StrOutputParser()

    final_answer = chain.invoke({
        "context": context,
        "question": answer_query
    })

    return final_answer


