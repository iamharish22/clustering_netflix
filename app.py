import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# -----------------------
# Load and preprocess data
# -----------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("C:/Users/OM/OneDrive/Projects/module 6 app/netflix_titles.csv")
    df = df.fillna("")
    df["combined"] = df["type"] + " " + df["listed_in"] + " " + df["description"]
    return df

df = load_and_prepare_data()

# -----------------------
# Create clusters
# -----------------------
@st.cache_resource
def cluster_data(df, num_clusters=10):
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(df["combined"])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)
    return df, kmeans

df, kmeans = cluster_data(df, num_clusters=10)

# -----------------------
# Recommendation function
# -----------------------
def recommend_from_cluster(title, n=5):
    title_row = df[df["title"].str.lower() == title.lower()]
    if title_row.empty:
        return []
    cluster_id = title_row["cluster"].values[0]
    cluster_titles = df[df["cluster"] == cluster_id]["title"]
    cluster_titles = cluster_titles[cluster_titles.str.lower() != title.lower()]
    return cluster_titles.sample(min(n, len(cluster_titles))).tolist()

# -----------------------
# Streamlit UI
# -----------------------
st.title("🎬 Netflix Cluster-based Recommender")
st.write("Select a title to get recommendations from the same cluster.")

# Select box for title
selected_title = st.selectbox(
    "Pick a Movie or TV Show:",
    sorted(df["title"].unique())
)

# Number of recommendations
num_recs = st.slider("Number of Recommendations:", 1, 10, 5)

# Show recommendations
if selected_title:
    recs = recommend_from_cluster(selected_title, n=num_recs)
    if recs:
        st.subheader("You might also like:")
        for r in recs:
            st.write(f"✅ {r}")
    else:
        st.warning("No recommendations found for this title.")