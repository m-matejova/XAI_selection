import streamlit as st
import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
from pymcdm.methods import TOPSIS
from pyDecision.algorithm import ahp_method
from pymcdm.weights import critic_weights

st.title("Selecting the XAI method using MCDM")

df = None
uploaded_file = st.sidebar.file_uploader("UPLOAD YOUR DATA", type=["xlsx", "csv"])
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    try:
        if file_extension == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif file_extension == "csv":
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")

metrics_num = st.sidebar.number_input("ENTER A NUMBER OF METRICS", min_value=1, max_value=30)

weights_form = st.sidebar.radio(
    "METHOD FOR CALCULATING WEIGHTS",
    ["Direct rating", "Pairwise comparison (subjective)", "CRITIC (objective)"], index=0
)

st.divider()

if df is not None:
    data = df.iloc[:, -metrics_num:]
    metric_name = data.columns.tolist()
    weights = []

    if weights_form == "Direct rating":
        metric_values = {}
        for i in metric_name:
            value = st.sidebar.slider(
                i,
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                key=f"metric_slider_{i}",
                help=f"Enter a weight for the {i}."
            )

            metric_values[i] = value

        weights_list = list(metric_values.values())
        weights = np.array(weights_list)
        weights = normalize([weights], norm="l1")
        weights = weights.flatten()

        st.subheader("Normalized weights:")

        result = pd.DataFrame({
            'Metric': metric_name,
            'Weight': np.round(weights, 4)
        })
        st.write(result)


    elif weights_form == "Pairwise comparison (subjective)":
        st.sidebar.write("## Saaty's pairwise comparison")
        if metrics_num >= 10:
            st.sidebar.write("A higher number of metrics is not recommended for this method.")
        else:
            # data = df.iloc[:, -metrics_num:]
            criteria = data.columns.tolist()

            comparison_matrix = np.ones((metrics_num, metrics_num))

            st.subheader("Calculated weights:")

            # User input for the matrix
            for i in range(metrics_num):
                for j in range(metrics_num):
                    if i == j:
                        comparison_matrix[i, j] = 1
                    elif i > j:  # Only show upper or lower triangle for input to avoid redundancy
                        comparison_matrix[i, j] = eval(
                            st.sidebar.selectbox(f"Compare {criteria[i]} to {criteria[j]}",
                                                 ("1", "3", "5", "7", "9", "1/3", "1/5", "1/7", "1/9")))
                        comparison_matrix[j, i] = 1 / comparison_matrix[i, j]

            weight_derivation = 'geometric'
            weights, rc = ahp_method(comparison_matrix, wd=weight_derivation)

            result = pd.DataFrame({
                'Metric': metric_name,
                'Weight': np.round(weights, 4)
            })
            st.write(result)

    else:  # weights_form == "CRITIC (objective)":
        criteria_types = np.ones(data.shape[1])

        weights = critic_weights(data, criteria_types)

        st.subheader("Calculated weights:")

        result = pd.DataFrame({
            'Metric': metric_name,
            'Weight': np.round(weights, 4)
        })

        st.write(result)

    if st.sidebar.button("Calculating preferences using TOPSIS", icon=":material/calculate:", key="filter",
                         width="stretch", type="primary"):

        weights = np.array(weights)
        types = np.ones(metrics_num)
        topsis = TOPSIS()

        pref_topsis = topsis(data, weights, types)

        df["Preference_TOPSIS"] = pref_topsis

        df = df.sort_values(by='Preference_TOPSIS', ascending=False)

        st.write('## Results')

        st.write(df)
