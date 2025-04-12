import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

# Page configuration
st.set_page_config(page_title="Amazon Review Insight", layout="wide")

# App Title
st.title("🛍️ Amazon Review Insight")
st.markdown("A tool to classify customer sentiment in Amazon product reviews.")

# Load model and tokenizer (will work once 'Model_BERT' is available)
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("Model_BERT")
    tokenizer = BertTokenizer.from_pretrained("Model_BERT")
    model.eval()
    return model, tokenizer

try:
    model, tokenizer = load_model()
    model_loaded = True
except Exception as e:
    st.error("❌ Model could not be loaded. Make sure 'Model_BERT' exists and contains the necessary files.")
    st.code(str(e))
    model_loaded = False

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
    category = st.selectbox("Select Product Category", ["All", "Fire & Amazon Tablets", "Smart Home & Entertainment Devices", "eBook Readers & Accessories"])
    min_rating = st.slider("Minimum Star Rating", 1, 5, 3)

# Example product categories and clusters
# Let's assume you already have the clusters defined and ready, here is a simple list of categories
clusters = ["Fire & Amazon Tablets", "Smart Home & Entertainment Devices", "eBook Readers & Accessories"]

# For the example, let's define some sample summaries for each cluster
cluster_summaries = {
    "Fire & Amazon Tablets": "Top Products: Expanding Accordion File Folder, AmazonBasics USB Cable.",
    "Smart Home & Entertainment Devices": "Top Products: Amazon Echo, Fire TV Stick.",
    "eBook Readers & Accessories": "Top Products: Kindle Paperwhite, Kindle Accessories."
}

# DataFrame Example - this would be your actual review dataset
data = {
    'meta_category': ['Fire & Amazon Tablets', 'Fire & Amazon Tablets', 'Smart Home & Entertainment Devices', 
                      'Smart Home & Entertainment Devices', 'eBook Readers & Accessories', 'eBook Readers & Accessories'],
    'product': ['Kindle Fire HD', 'Fire Stick', 'Amazon Echo', 'Fire TV Stick', 'Kindle Paperwhite', 'Kindle Case'],
    'review': ['Excellent tablet!', 'Great streaming device!', 'Amazing smart home assistant!', 'Super easy to use!', 'Great for reading!', 'Good protection for Kindle!'],
    'rating': [5, 4, 5, 4, 5, 4]
}
df = pd.DataFrame(data)

# --- Sentiment Classifier Tab ---
tab1, tab2 = st.tabs(["📊 Sentiment Classifier", "📝 Review Summary"])

with tab1:
    st.subheader("🗣️ Enter a Product Review")
    review = st.text_area("Paste the review text below:", height=150)

    if st.button("🔍 Classify Sentiment"):
        if not model_loaded:
            st.warning("⚠️ Cannot classify because the model is not loaded.")
        elif review.strip() == "":
            st.warning("Please enter a review.")
        else:
            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

            sentiment_map = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😄"}
            color_map = {0: "red", 1: "gray", 2: "green"}

            sentiment = sentiment_map[pred]
            st.markdown(
                f"<h3 style='color:{color_map[pred]};'>Predicted Sentiment: {sentiment}</h3>",
                unsafe_allow_html=True,
            )

# --- Review Summary Tab ---
with tab2:
    st.subheader("📦 Category-Based Review Summary")
    st.info(f"Selected Category: **{category}** | Minimum Rating: **{min_rating} stars**")
    
    if category != "All":
        st.write(f"📌 **{category}** - Showing reviews with rating **{min_rating}** stars and above.")
        st.write("Here's a quick summary of the top-rated products and some key insights:")

        # Filter reviews based on selected category and minimum rating
        df_filtered = df[(df['meta_category'] == category) & (df['rating'] >= min_rating)]
        st.write(df_filtered[['product', 'review', 'rating']])

        # Example of a cluster summary that matches the selected category
        if category in cluster_summaries:
            st.markdown(f"**Cluster Summary for {category}:**")
            st.write(cluster_summaries[category])

    else:
        st.write("Please select a category to view the summary.")

    # Example of review display
    st.image("https://images-na.ssl-images-amazon.com/images/I/61v2zDdGpaL._AC_SL1000_.jpg", width=200)
    st.markdown("""
    **Top Product 1:** Kindle Paperwhite  
    - Long battery life  
    - Sharp display  
    - Waterproof
    """)
