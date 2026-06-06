import streamlit as st
import pandas as pd

st.sidebar.title("Project Information")

st.sidebar.info(
    """
    Technologies Used:

    • Python
    • Scikit-learn
    • Pandas
    • Streamlit
    • NLP
    """
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv("reviews.csv")
data = data.dropna()

# Train model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

model.fit(data["review"], data["sentiment"])

# Page title
st.title("🛍️ AI-Powered Product Review Sentiment Analysis")

st.markdown(
    "Analyze customer feedback using Machine Learning and Natural Language Processing."
)
uploaded_file = st.file_uploader(
    "Upload a CSV file of reviews",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Reviews")

    df["Predicted Sentiment"] = model.predict(df["review"])

    st.dataframe(df)

    st.subheader("Sentiment Summary")

    summary = df["Predicted Sentiment"].value_counts()

    st.bar_chart(summary)

    st.write(summary)


# User input
review = st.text_area("Enter a Product Review")

# Predict
if st.button("Analyze Sentiment"):

    prediction = model.predict([review])[0]

    if prediction == "positive":
        st.success(f"😊 Sentiment: {prediction.title()}")

    elif prediction == "negative":
        st.error(f"😞 Sentiment: {prediction.title()}")

    else:
        st.warning(f"😐 Sentiment: {prediction.title()}")