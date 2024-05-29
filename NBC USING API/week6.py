import streamlit as st
import pandas as pd
import numpy as np

# Streamlit interface
st.title("Sentiment Analysis with Naive Bayes")

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    msg = pd.read_csv(uploaded_file, names=['message', 'label'])
    st.write("Total Instances of Dataset: ", msg.shape[0])

    msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

    X = msg.message
    y = msg.labelnum

    # Split data into training and testing sets
    train_size = int(0.8 * len(X))
    Xtrain, Xtest = X[:train_size], X[train_size:]
    ytrain, ytest = y[:train_size], y[train_size:]

    # Tokenization
    words = ' '.join(Xtrain).split()
    vocabulary = list(set(words))
    vocab_len = len(vocabulary)

    # Count words for each class
    word_counts_pos = np.zeros(vocab_len)
    word_counts_neg = np.zeros(vocab_len)

    for message, label in zip(Xtrain, ytrain):
        for word in message.split():
            if label == 1:
                word_counts_pos[vocabulary.index(word)] += 1
            else:
                word_counts_neg[vocabulary.index(word)] += 1

    # Calculate probabilities
    prior_pos = sum(ytrain) / len(ytrain)
    prior_neg = 1 - prior_pos

    word_probs_pos = (word_counts_pos + 1) / (sum(word_counts_pos) + vocab_len)
    word_probs_neg = (word_counts_neg + 1) / (sum(word_counts_neg) + vocab_len)

    # Predict function
    def predict(message):
        pos_score = np.log(prior_pos)
        neg_score = np.log(prior_neg)
        for word in message.split():
            if word in vocabulary:
                pos_score += np.log(word_probs_pos[vocabulary.index(word)])
                neg_score += np.log(word_probs_neg[vocabulary.index(word)])
        return 1 if pos_score > neg_score else 0

    # Predictions
    predictions = [predict(message) for message in Xtest]

    # Display predictions
    st.write("Predictions:")
    for doc, p in zip(Xtest, predictions):
        p = 'pos' if p == 1 else 'neg'
        st.write("%s -> %s" % (doc, p))

    # Calculate accuracy
    accuracy = sum(predictions == ytest) / len(ytest)
    st.write("Accuracy: ", accuracy)
