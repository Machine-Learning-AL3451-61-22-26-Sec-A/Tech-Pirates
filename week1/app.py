import numpy as np
import pandas as pd
data = pd.DataFrame(data=pd.read_csv("C:\\Users\\ashik\\OneDrive\\Desktop\\CEA\\trainingdata.csv"))
import streamlit as st

print(data)
concepts = np.array(data.iloc[:,0:-1])
print(concepts)
target = np.array(data.iloc[:,-1])
print(target)

def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("\nInitialization of specific_h and general_h")
    print(specific_h)
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print(general_h)
   
    for i, h in enumerate(concepts):
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        if target[i] == "No":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("\nSteps of Candidate Elimination Algorithm",i+1)
        print(specific_h)
        print(general_h)

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print("\nFinal Specific_h:", s_final, sep="\n")
print("\nFinal General_h:", g_final, sep="\n")

def main():
    st.title('Candidate Elimination Algorithm')
    st.write("Upload your training data (CSV file):")
    uploaded_file = st.file_uploader("C:\\Users\\ashik\\OneDrive\\Desktop\\CEA\\trainingdata.csv", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Training Data:")
        st.write(data)
        concepts = np.array(data.iloc[:, 0:-1])
        target = np.array(data.iloc[:, -1])
        s_final, g_final = learn(concepts, target)
        st.write("Final Specific_h:")
        st.write(s_final)
        st.write("Final General_h:")
        st.write(g_final)

if __name__ == "__main__":
    main()