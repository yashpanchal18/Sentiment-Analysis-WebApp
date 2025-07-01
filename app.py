import os
import pandas as pd
import streamlit as st
import joblib
import psycopg2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import altair as alt
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import plotly.express as px
import geopandas as gpd
import time
import nltk
from tensorflow.keras.models import load_model
from preprocess import review_to_token_vectors
from hybrid import combine_explanations, shap_explain, lime_explain
import numpy as np
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
# ----------------------------- Streamlit Setup -----------------------------
st.set_page_config(
    page_title="Product Reviews",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")
col = st.columns((1.5, 4.5, 2), gap='medium')

# ----------------------------- Load Model -----------------------------
# Load attention model + encoder
model = load_model("models/lstm_attention_model.h5")
label_encoder = joblib.load("models/lstm_attention_label_encoder.pkl")

# ----------------------------- PostgreSQL Connection -----------------------------
def create_connection():
    conn = psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        database=st.secrets["postgres"]["database"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        port=st.secrets["postgres"]["port"]
    )
    return conn

def create_table_if_not_exists():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_reviews (
            id SERIAL PRIMARY KEY,
            cid VARCHAR(50),
            country VARCHAR(50),
            product VARCHAR(50),
            review TEXT,
            sentiment VARCHAR(20)
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

create_table_if_not_exists()

def insert_review(cid, country, product, review, sentiment):
    conn = create_connection()
    cursor = conn.cursor()
    insert_query = """
        INSERT INTO customer_reviews (cid, country, product, review, sentiment)
        VALUES (%s, %s, %s, %s, %s);
    """
    cursor.execute(insert_query, (cid, country, product, review, sentiment))
    conn.commit()
    cursor.close()
    conn.close()

def fetch_reviews():
    conn = create_connection()
    df = pd.read_sql_query("""
        SELECT 
            cid AS "CiD",
            country AS "Country",
            product AS "Product",
            review AS "Review",
            sentiment AS "Sentiment"
        FROM customer_reviews
    """, conn)
    conn.close()
    return df

# ----------------------------- Cleaning and Prediction -----------------------------
def get_confidence(text):
    vec = review_to_token_vectors(clean_text(text), max_len=30)
    pred = model.predict(np.array([vec]), verbose=0)[0]
    return max(pred)

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = review_to_token_vectors(cleaned, max_len=30)
    prediction = model.predict(np.array([vec]), verbose=0)[0]
    label = np.argmax(prediction)
    sentiment_dict = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_dict[label]


# ----------------------------- Sidebar Menu -----------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Add Review", "Dashboard"],
        default_index=0,
    )

# ----------------------------- Add Review Section -----------------------------
if selected == "Add Review":
    st.title('Product Sentiment Analysis')

    cust_id = st.text_input('Customer ID')
    countries = ["Australia", "Austria", "Belgium", "Brazil", "China", "India", "Japan", "Zambia"]
    country = st.selectbox("Choose your country", countries)

    if country:
        product = ["Monitor", "PC", "Laptop", "T-Shirt", "Cutlery", "Guitar", "Curtains"]
        select = st.selectbox("Choose the product", product)
        review = st.text_area('Enter your thoughts about the product!')

        if review:
            click = st.button('Submit')
            if click:
                cleaned = clean_text(review)
                confidence = get_confidence(review)

                lime_scores = lime_explain(review, top_k=10)
                shap_scores = shap_explain(review, top_k=10)
                hybrid_scores = combine_explanations(lime_scores, shap_scores, strategy="auto", model_confidence=confidence)

                predicted_sentiment = predict_sentiment(review)
                insert_review(cust_id, country, select, review, predicted_sentiment)

                st.success('Your review was successfully recorded!')
                st.write("Predicted Sentiment:", predicted_sentiment)

                # Highlight important tokens
                tokens = review.split()
                important_tokens = {tok.lower(): val for tok, val in hybrid_scores.items() if abs(val) >= 0.01}

                highlighted = " ".join(
                [f"**:blue[{t}]**" if t.lower() in important_tokens else t for t in tokens]
                )
                st.markdown("### 🔍 Key Influential Words in Review:")
                st.markdown(highlighted)


# ----------------------------- Dashboard Section -----------------------------
if selected == "Dashboard":
    with st.spinner('Loading Dashboard... Please wait'):
        df = fetch_reviews()
        time.sleep(1)

    # Sidebar Filters
    st.sidebar.markdown('---')
    st.sidebar.header('📊 Filters')
    countries = ['All'] + sorted(df['Country'].dropna().unique())
    products = ['All'] + sorted(df['Product'].dropna().unique())
    sentiments = ['All'] + sorted(df['Sentiment'].dropna().unique())

    selected_country = st.sidebar.selectbox("Select Country", countries)
    selected_product = st.sidebar.selectbox("Select Product", products)
    selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiments)

    # Apply Filters
    filtered_df = df.copy()
    if selected_country != 'All':
        filtered_df = filtered_df[filtered_df['Country'] == selected_country]
    if selected_product != 'All':
        filtered_df = filtered_df[filtered_df['Product'] == selected_product]
    if selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['Sentiment'] == selected_sentiment]

    # ======================= KPI Section =======================
    st.markdown("### 🔢 Key Metrics")
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        total_customers = filtered_df['CiD'].nunique()
        st.metric("👥 Total Customers", total_customers)
    with kpi2:
        total_reviews = filtered_df.shape[0]
        st.metric("📝 Total Reviews", total_reviews)
    with kpi3:
        avg_reviews = round(total_reviews / total_customers, 2) if total_customers else 0
        st.metric("📊 Avg. Reviews per Customer", avg_reviews)

    st.markdown("---")

    # ======================= Middle Row (Two Columns) =======================
    left, right = st.columns(2)

    # ---- Left Column Visuals ----
    with left:
        st.markdown("### 📌 Sentiment Distribution")
        sentiment_counts = filtered_df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        if not sentiment_counts.empty:
            pie_chart = alt.Chart(sentiment_counts).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Sentiment", type="nominal"),
                tooltip=['Sentiment', 'Count']
            ).properties(width=350, height=300)
            st.altair_chart(pie_chart, use_container_width=True)
        else:
            st.info("No sentiment data available.")

        st.markdown("### 🌍 Top 5 Countries by Review Count")
        top_countries = filtered_df['Country'].value_counts().reset_index()
        top_countries.columns = ['Country', 'Review Count']
        st.dataframe(top_countries.head(5))

    # ---- Right Column Visuals ----
    with right:
        st.markdown("### 🗺️ Reviews by Country (Map)")
        country_counts = filtered_df['Country'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Review Count']
        if not country_counts.empty:
            fig_map = px.choropleth(
                country_counts,
                locations="Country",
                locationmode='country names',
                color="Review Count",
                hover_name="Country",
                color_continuous_scale=px.colors.sequential.Plasma
            )
            fig_map.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No country data to display.")

        st.markdown("### 📍 Customers per Country")
        country_customer_counts = filtered_df.groupby('Country')['CiD'].nunique().reset_index()
        country_customer_counts.columns = ['Country', 'Total Customers']
        if not country_customer_counts.empty:
            bar_chart_country = alt.Chart(country_customer_counts).mark_bar().encode(
                x=alt.X('Total Customers:Q'),
                y=alt.Y('Country:N', sort='-x'),
                tooltip=['Country', 'Total Customers']
            ).properties(height=300)
            st.altair_chart(bar_chart_country, use_container_width=True)
        else:
            st.info("No customer data to show.")

    st.markdown("---")

    # ======================= Bottom Row =======================
    st.markdown("### 📦 Product Sentiment Breakdown")
    product_sentiment = filtered_df.groupby(['Product', 'Sentiment']).size().reset_index(name='Count')
    product_total = filtered_df['Product'].value_counts().reset_index()
    product_total.columns = ['Product', 'Total']
    product_sentiment = pd.merge(product_sentiment, product_total, on='Product')
    product_sentiment['Percentage'] = (product_sentiment['Count'] / product_sentiment['Total']) * 100

    if not product_sentiment.empty:
        stacked_bar = alt.Chart(product_sentiment).mark_bar().encode(
            x=alt.X('Percentage:Q', stack='normalize'),
            y=alt.Y('Product:N', sort='-x'),
            color='Sentiment:N',
            tooltip=['Product', 'Sentiment', 'Percentage']
        ).properties(height=400)
        st.altair_chart(stacked_bar, use_container_width=True)
    else:
        st.info("No product sentiment data available.")

    st.markdown("### 🎯 Product Highlights")
    col1, col2 = st.columns(2)
    with col1:
        best = filtered_df[filtered_df['Sentiment'] == 'Positive']['Product'].value_counts()
        if not best.empty:
            st.success(f"⭐ Best Product: {best.idxmax()}")
    with col2:
        worst = filtered_df[filtered_df['Sentiment'] == 'Negative']['Product'].value_counts()
        if not worst.empty:
            st.error(f"⚠️ Worst Product: {worst.idxmax()}")

    with st.expander("ℹ️ About this Dashboard", expanded=False):
        st.write("""
        - **Total Customers**: Unique customer IDs submitting reviews.
        - **Total Reviews**: All customer feedback entries.
        - **Sentiment Distribution**: Pie breakdown of positive, neutral, negative.
        - **Country Metrics**: Visuals on which countries are contributing the most.
        - **Product Sentiment**: Breakdown by sentiment and product.
        """)
