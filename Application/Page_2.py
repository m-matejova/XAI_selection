import streamlit as st
import pandas as pd

dff = pd.read_excel("Filtering all.xlsx")

st.title("Filtering methods")

st.divider()

st.sidebar.write("# Filtering")


scope_choice = st.sidebar.pills('SCOPE', ["Local", "Global"])

portability_choice = st.sidebar.pills('PORTABILITY', ["Model-agnostic", "Model-specific"])
if portability_choice == "Model-specific":
    model_choice = st.sidebar.pills('TYPE OF MODEL', ["Tree ensemble", "Neural network", "SVM", "Other"])

problem_choice = st.sidebar.pills('TYPE OF PROBLEM', ["Classification", "Regression"])

data_choice = st.sidebar.pills('INPUT DATA TYPE', ["TAB", "IMG", "TXT", "GRH", "TS", "VID"])

output_choice = st.sidebar.pills('OUTPUT FORMAT', ["Visual", "Textual", "Numerical", "Rules"])


if st.sidebar.button("Filter", icon=":material/search:", key="filter", width="stretch", type="primary"):
    if pd.notna(portability_choice):
        dff = dff.loc[(dff['Portability'] == portability_choice)]
    if pd.notna(scope_choice):
        dff = dff.loc[(dff['Scope'] == scope_choice)]
    if pd.notna(data_choice):
        dff = dff[dff['Input data type'].str.contains(data_choice)]
    if pd.notna(problem_choice):
        dff = dff[dff['Problem'].str.contains(problem_choice)]
    if pd.notna(output_choice):
        dff = dff[dff['Output'].str.contains(output_choice)]

num_methods = len(dff)
st.write("Number of filtered methods: ", num_methods)
st.write(dff)