import streamlit as st
from langchain.llms import HuggingFaceHub
from huggingface_hub import InferenceApi
from langchain.chains import RetrievalQA
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI


def load_huggingface_model():
    huggingface_api = st.secrets["API_TOKEN"]

    # Initialize the HuggingFaceHub model
    model_name = "mistralai/Mistral-7B-v0.1"
    model_2 = "Qwen/Qwen2.5-1.5B"
    inference = InferenceApi(repo_id=model_2, token=huggingface_api)

    model = HuggingFaceHub(
        huggingfacehub_api_token=huggingface_api,
        repo_id=model_name,
        model_kwargs={"temperature": 0.6, "max_new_tokens": 50}
    )

    return model


# def load_gemini_model():
#     gemini_api = st.secrets["GEMINI_API_KEY"]
#     genai.configure(api_key="AIzaSyDz31p7uHa_CjmhWPZYHPpspvmvVPQUsfY")

#     geminiLLM = ChatGoogleGenerativeAI(
#         model="gemini-1.5-pro",
#         temperature=0.6,
#         max_tokens=500,
#         timeout=None,
#         max_retries=2,
#     )
#     return geminiLLM


def initialize_qa_chain(retriever):

    model = load_huggingface_model()
    # gemini_model = load_gemini_model()

    # Setting up the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=model, retriever=retriever, chain_type="stuff", return_source_documents=False)

    return qa_chain
