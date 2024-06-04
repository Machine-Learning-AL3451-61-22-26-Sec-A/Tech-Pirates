import pandas as pd
import numpy as np

# Function to prompt user for binary input
def get_binary_input(prompt):
    while True:
        response = input(prompt + " (Enter 'yes' or 'no'): ").strip().lower()
        if response in {'yes', 'no'}:
            return 1 if response == 'yes' else 0
        else:
            print("Invalid input! Please enter 'yes' or 'no'.")

# Function to prompt user for numerical input
def get_numerical_input(prompt):
    while True:
        response = input(prompt + ": ").strip()
        if response.isdigit():
            return int(response)
        else:
            print("Invalid input! Please enter a numerical value.")

# Function to calculate probability of having Corona based on input
def calculate_probability(evidence):
    # Simple mock-up probabilities for demonstration purposes
    # In a real-world scenario, these should be derived from a trained Bayesian network
    base_prob = 0.05  # Base probability of having Corona without any evidence
    if evidence['Corona_test'] == 1:
        return 0.9  # High probability if tested positive

    # Adjust the probability based on evidence (This is a simplified example)
    prob = base_prob
    if evidence['Fever'] == 1:
        prob += 0.1
    if evidence['Cough'] == 1:
        prob += 0.1
    if evidence['Breathlessness'] == 1:
        prob += 0.15
    if evidence['Travel_history'] == 1:
        prob += 0.1
    if evidence['Age'] > 60:
        prob += 0.1

    return min(prob, 1.0)  # Ensure the probability does not exceed 1

# Main function to run the script
def main():
    print("Bayesian Network for Coronavirus Diagnosis")

    # Collect evidence from user input
    evidence = {
        'Fever': get_binary_input("Do you have fever?"),
        'Cough': get_binary_input("Do you have cough?"),
        'Breathlessness': get_binary_input("Do you experience breathlessness?"),
        'Corona_test': get_binary_input("Have you taken a coronavirus test?"),
        'Travel_history': get_binary_input("Have you traveled recently?"),
        'Age': get_numerical_input("What is your age?")
    }

    # Calculate the probability of having Corona
    prob_corona = calculate_probability(evidence)
    print(f"\nProbability of having Coronavirus: {prob_corona * 100:.2f}%")

if __name__ == "__main__":
    main()
