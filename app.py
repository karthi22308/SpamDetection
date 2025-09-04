import streamlit as st
import pickle
import string
import csv
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import warnings
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Initialize models
tfidf = None
model = None

# Try to load pre-trained models, otherwise create and fit new ones
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    st.success("✅ Loaded pre-trained models successfully!")
    
    # Test if the vectorizer is fitted
    try:
        # This will raise an error if not fitted
        tfidf.transform(["test message"])
    except Exception as e:
        st.warning("Vectorizer found but not fitted. Creating new models...")
        raise Exception("Vectorizer not fitted")
        
except (FileNotFoundError, Exception) as e:
    st.warning("Creating new models with sample training data...")
    
    # Comprehensive sample training data
    spam_samples = [
        "free money win lottery prize", "urgent bank account update required",
        "congratulations you won iphone", "click link claim your reward",
        "discount sale limited time offer", "hurry buy now 50% off",
        "job vacancy apply now tcs", "government railway jobs available",
        "you have won $1000", "claim your free gift now",
        "bank security alert verify account", "password reset required immediately",
        "limited time offer buy one get one free", "exclusive deal for you",
        "earn money from home no experience", "get rich quick easy money",
        "medical bill refund available", "inheritance money waiting for you",
        "lottery winner announcement", "credit card offer pre-approved"
    ]
    
    ham_samples = [
        "meeting tomorrow at 10am", "project update from team",
        "lunch plans for friday", "weekly report attached",
        "hello how are you doing", "just checking in on you",
        "family dinner this weekend", "birthday party invitation",
        "work schedule for next week", "team building activity",
        "project deadline extension", "client meeting rescheduled",
        "thanks for your help yesterday", "appreciation for your work",
        "weekend plans discussion", "weather forecast for tomorrow",
        "office holiday schedule", "team lunch next monday",
        "project status update meeting", "weekly progress report"
    ]
    
    # Combine samples and labels
    sample_texts = spam_samples + ham_samples
    sample_labels = [1] * len(spam_samples) + [0] * len(ham_samples)  # 1=spam, 0=ham
    
    # Transform sample texts
    transformed_texts = [transform_text(text) for text in sample_texts]
    
    # Create and fit TF-IDF vectorizer with proper parameters
    tfidf = TfidfVectorizer(
        max_features=5000,
        min_df=1,
        max_df=0.8,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit the vectorizer
    X = tfidf.fit_transform(transformed_texts)
    
    # Train a model
    model = MultinomialNB()  # Using Naive Bayes which often works better for text
    model.fit(X, sample_labels)
    
    # Save the models for future use
    try:
        pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
        pickle.dump(model, open('model.pkl', 'wb'))
        st.info("💾 Models created and saved for future use")
    except Exception as save_error:
        st.warning(f"Could not save models: {save_error}")
    
    st.success("✅ Models created and trained successfully!")

# Define categories
sports = ['basketball', 'football', 'tennis', 'cricket', 'sport', 'game', 'match', 'player', 'team']
offers = ['discount', 'sale', 'hurry', 'limited', 'offer', 'deal', 'promotion', 'coupon', 'special']
shopping = ['flipkart', 'amazon', 'ebay', 'meesho', 'shop', 'buy', 'purchase', 'cart', 'product']
jobalert = ['vacancy', 'job', 'tcs', 'wipro', 'government', 'railway', 'public', 'career', 'hire', 'employment']

def check_categories(text, options):
    """Check if text matches any of the selected categories"""
    if not options:
        return False
        
    text_lower = text.lower()
    
    if 'sports' in options:
        for sport in sports:
            if re.search(r'\b' + re.escape(sport) + r'\b', text_lower):
                return True
    
    if 'offers' in options:
        for offer in offers:
            if re.search(r'\b' + re.escape(offer) + r'\b', text_lower):
                return True
    
    if 'e-shopping' in options:
        for eshop in shopping:
            if re.search(r'\b' + re.escape(eshop) + r'\b', text_lower):
                return True
    
    if 'job alert' in options:
        for job in jobalert:
            if re.search(r'\b' + re.escape(job) + r'\b', text_lower):
                return True
    
    return False

def predict_email(text):
    """Predict if email is malicious with error handling"""
    try:
        transformed_sms = transform_text(text)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        return result
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

st.title("📧 Malicious Email Detector")

# Spam customizer
mde = st.radio("Spam customizer", ('disable', 'enable'), horizontal=True)
options = []
opt = 0

if mde == "enable":
    options = st.multiselect(
        'Select categories to monitor:',
        ['sports', 'offers', 'e-shopping', 'job alert'],
        ['sports']
    )
    opt = 1

# Main mode selection
cn = st.radio("Select Mode", ('TEST', 'INBOX'), horizontal=True)

if cn == "TEST":
    mode = st.radio("Select Input Mode", ('Single Message', 'Multiple Messages'), horizontal=True)
    
    if mode == "Single Message":
        input_sms = st.text_area("Enter the email received", height=150, placeholder="Paste your email content here...")
        
        if st.button('🔍 Analyze Email', type="primary"):
            if not input_sms.strip():
                st.warning("Please enter a message to analyze.")
            else:
                with st.spinner("Analyzing message..."):
                    result = predict_email(input_sms)
                    
                    if result is not None:
                        if result == 1:
                            st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 32px;">⚠️ Warning! Malicious email detected</p>', unsafe_allow_html=True)
                            
                            if opt == 1 and options:
                                if check_categories(input_sms, options):
                                    st.markdown('<p style="font-family:sans-serif; color:Orange; font-size: 24px;">🔔 Related to your monitored interests</p>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p style="font-family:sans-serif; color:Green; font-size: 32px;">✅ Safe Email</p>', unsafe_allow_html=True)
    
    else:  # Multiple Messages
        paragraph = st.text_area("Enter emails separated by double space (  )", height=200,
                               placeholder="Email 1  Email 2  Email 3",
                               help="Separate each email with double spaces")
        
        if st.button('🔍 Analyze Multiple Emails', type="primary"):
            if not paragraph.strip():
                st.warning("Please enter messages to analyze.")
            else:
                with st.spinner("Analyzing multiple messages..."):
                    sentences = [s.strip() for s in paragraph.split('  ') if s.strip()]
                    
                    if not sentences:
                        st.warning("No valid messages found.")
                    else:
                        results = []
                        interests = []
                        
                        for sentence in sentences:
                            result = predict_email(sentence)
                            if result is not None:
                                if result == 1:
                                    results.append("Malicious Email")
                                    if opt == 1 and options:
                                        if check_categories(sentence, options):
                                            interests.append("Related to Interests")
                                        else:
                                            interests.append("Not Related")
                                    else:
                                        interests.append("N/A")
                                else:
                                    results.append("Safe Email")
                                    interests.append("N/A")
                            else:
                                results.append("Error")
                                interests.append("N/A")
                        
                        # Create DataFrame for display
                        df_data = {'Message': sentences, 'Result': results}
                        if opt == 1:
                            df_data['Interest'] = interests
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Statistics
                        malicious_count = sum(1 for r in results if "Malicious" in r)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Emails", len(sentences))
                        with col2:
                            st.metric("Malicious Emails", malicious_count)

elif cn == "INBOX":
    st.title("📥 INBOX Analysis")
    
    # File upload option
    uploaded_file = st.file_uploader("Upload inbox CSV file", type=['csv'], 
                                   help="CSV file should contain email messages in the first column")
    
    if uploaded_file is not None:
        try:
            df_inbox = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df_inbox)} emails")
            
            # Display first few rows
            st.write("Preview of your inbox:")
            st.dataframe(df_inbox.head(), use_container_width=True)
            
            # Assume first column contains messages
            message_column = df_inbox.columns[0]
            
            if st.button('📊 Analyze Inbox', type="primary"):
                with st.spinner("Analyzing inbox emails..."):
                    results = []
                    interests = []
                    processed_count = 0
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, message in enumerate(df_inbox[message_column]):
                        status_text.text(f"Processing email {i+1}/{len(df_inbox)}...")
                        progress_bar.progress((i + 1) / len(df_inbox))
                        
                        try:
                            message_str = str(message)
                            result = predict_email(message_str)
                            
                            if result is not None:
                                if result == 1:
                                    results.append("Malicious Email")
                                    if opt == 1 and options:
                                        if check_categories(message_str, options):
                                            interests.append("Related to Interests")
                                        else:
                                            interests.append("Not Related")
                                    else:
                                        interests.append("N/A")
                                else:
                                    results.append("Safe Email")
                                    interests.append("N/A")
                            else:
                                results.append("Error")
                                interests.append("N/A")
                                
                            processed_count += 1
                                
                        except Exception as e:
                            results.append(f"Error: {str(e)}")
                            interests.append("N/A")
                    
                    status_text.text("Analysis complete!")
                    progress_bar.empty()
                    
                    # Add results to dataframe
                    df_results = df_inbox.copy()
                    df_results['Analysis_Result'] = results
                    if opt == 1:
                        df_results['Interest_Category'] = interests
                    
                    st.success(f"✅ Analysis completed! Processed {processed_count} emails")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Show statistics
                    malicious_count = sum(1 for r in results if "Malicious" in r)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Emails", len(df_inbox))
                    with col2:
                        st.metric("Malicious Emails", malicious_count)
                    with col3:
                        st.metric("Safe Emails", len(df_inbox) - malicious_count)
                    
                    # Download option
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Analysis Results",
                        data=csv,
                        file_name="inbox_analysis_results.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("📋 Please upload a CSV file containing your inbox emails to begin analysis.")

# Footer
st.markdown("---")
st.caption("🔒 Email Security Analyzer | Uses machine learning to detect malicious emails")