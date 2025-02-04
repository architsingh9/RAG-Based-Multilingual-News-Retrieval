import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from bert_score import score
from sklearn.metrics.pairwise import cosine_similarity

# Load SentenceTransformer model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Load data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Compute embeddings
# def generate_embeddings(_model, text_data):
#     """Generate embeddings for the provided text data."""
#     return np.array([_model.encode(text, normalize_embeddings=True) for text in text_data])
@st.cache_data
def generate_embeddings(_model, df):
    df['embedding'] = df['translated_text'].apply(
        lambda x: _model.encode(x, normalize_embeddings=True)
    )
    return np.vstack(df['embedding'].values)

# Create FAISS index
@st.cache_resource
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Encode query
def encode_query(query, model):
    return model.encode(query, normalize_embeddings=True)

# Perform FAISS search
def search_faiss(query_embedding, faiss_index, top_k=5):
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)
    return indices[0], distances[0]

# Compute BERTScore
def compute_bertscore(references, candidates, lang="en"):
    P, R, F1 = score(references, candidates, lang=lang, model_type="bert-base-uncased")
    return F1

# Streamlit App
st.title("Multilingual News Retrieval and Similarity Analysis")

# Step 1: Load Dataset
st.sidebar.header("Configuration")
data_path = st.sidebar.text_input(
    "Dataset Path",
    "dataset.csv"
)

if data_path:
    df = load_data(data_path)
    # st.write("### Dataset Preview", df.head())
    
    # Step 2: Generate Embeddings
    embedding_model = load_embedding_model()
    # df['embedding'] = generate_embeddings(embedding_model, df['translated_text'])
    # embeddings = np.vstack(df['embedding'])
    embeddings = generate_embeddings( embedding_model, df)
    st.write("Embeddings generated.")

    # Step 3: Build FAISS Index
    faiss_index = create_faiss_index(embeddings)
    st.sidebar.success(f"FAISS index contains {faiss_index.ntotal} articles.")
    
    # Step 4: User Query Input
    user_query = st.text_input("Enter your search query:")
    if user_query:
        query_embedding = encode_query(user_query, embedding_model)

        # Step 5: Perform FAISS Search
        top_k = st.sidebar.slider("Number of Results", min_value=1, max_value=10, value=5)
        indices, distances = search_faiss(query_embedding, faiss_index, top_k)
        retrieved_articles = df.iloc[indices].copy()
        
        # Step 6: Compute Cosine Similarity
        retrieved_embeddings = embeddings[indices]
        cosine_similarities = cosine_similarity([query_embedding], retrieved_embeddings).flatten()
        retrieved_articles['similarity'] = cosine_similarities

        # Step 7: Compute BERTScore
        query_list = [user_query] * len(retrieved_articles)
        retrieved_articles['query_summary_similarity'] = compute_bertscore(
            retrieved_articles["summary"].tolist(), query_list
        ).tolist()
        retrieved_articles['news_summary_similarity'] = compute_bertscore(
            retrieved_articles["summary"].tolist(), retrieved_articles["translated_text"].tolist()
        ).tolist()

        # Step 8: Display Results
        st.write("### Retrieved Articles with Similarity Scores")
        st.dataframe(
            retrieved_articles[
                ["title", "topic", "translated_text", "summary", "similarity", "query_summary_similarity", "news_summary_similarity"]
            ]
        )