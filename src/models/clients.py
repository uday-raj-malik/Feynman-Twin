import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder

@st.cache_resource
def get_embedding_model():
    """
    Loads and caches the BAAI/bge-small-en-v1.5 sentence transformer.
    This model embeds text in a 384-dimensional normalized vector space.
    """
    return SentenceTransformer('BAAI/bge-small-en-v1.5')

@st.cache_resource
def get_reranker_model():
    """
    Loads and caches the BAAI/bge-reranker-base CrossEncoder reranker.
    Used for query-document pair scoring to find the top 5 relevant windows.
    """
    return CrossEncoder('BAAI/bge-reranker-base')
