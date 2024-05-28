import streamlit as st
import pandas as pd

def load_data():
    try:
        return pd.read_csv('C:\\Users\\ashik\\OneDrive\\Desktop\\week6\\corona.csv')
    except FileNotFoundError:
        st.error("Error: Dataset file not found.")
        return None

def preprocess_data(data):
    # Check if data is not None
    if data is not None:
        # Drop any rows with missing values
        data.dropna(inplace=True)
    return data

def main():
    st.title("CORONA Infection Diagnosis")
    
    # Load data
    data = load_data()
    st.subheader("Dataset")
    st.write(data)
    
    # Preprocess data
    data = preprocess_data(data)
    st.subheader("Preprocessed Dataset")
    st.write(data)
    
    # User input for symptoms
    st.subheader("Enter Symptoms")
    fever = st.checkbox("Fever")
    cough = st.checkbox("Cough")
    breathlessness = st.checkbox("Shortness of Breath")
    fatigue = st.checkbox("Fatigue")
    body_aches = st.checkbox("Body Aches")
    loss_of_taste_smell = st.checkbox("Loss of Taste/Smell")
    
    # Analyze symptoms
    if fever or cough or breathlessness or fatigue or body_aches or loss_of_taste_smell:
        st.subheader("Diagnosis")
        st.write("Based on the symptoms entered, please consult a healthcare professional for further evaluation.")
    else:
        st.subheader("Diagnosis")
        st.write("No symptoms selected. Please select at least one symptom.")

if __name__ == "__main__":
    main()
