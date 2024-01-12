import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Function to load data
@st.cache(allow_output_mutation=True)
def load_data():
    data_path = '/Users/pranjalkaushik/Downloads/CompanyNewsHeadlines_ICICI_Consolidated.xls'
    try:
        return pd.read_excel(data_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Function to predict sentiment
def predict_sentiment(model, vectorizer, label_encoder, new_texts):
    new_data_vectorized = vectorizer.transform(new_texts)
    new_predictions = model.predict(new_data_vectorized)
    new_predictions_labels = label_encoder.inverse_transform(new_predictions)
    return new_predictions_labels

# Streamlit app
def main():
    st.title("ICICI Bank News Headlines Sentiment Analysis")

    # Load and preprocess the data
    df = load_data()
    if df.empty:
        st.warning("No data to display.")
        return

    X = df['tweet_text']
    y = df['Azure label']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vectorized, y_train)

    # User input
    user_input = st.text_area("Enter a news headline for sentiment prediction:", "")
    if user_input:
        # Predict and display sentiment
        predicted_sentiment = predict_sentiment(model, vectorizer, label_encoder, [user_input])
        st.write("Predicted Sentiment:", predicted_sentiment[0])

    # Display model accuracy on test data
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Model Accuracy:", accuracy)

    # Display classification report
    st.text("Classification Report:")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    st.json(report)

if __name__ == "__main__":
    main()
