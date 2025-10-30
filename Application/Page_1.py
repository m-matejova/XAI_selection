import streamlit as st

st.title(" 📶 A Multi-Criteria Decision Making Approach for the Selection of Explainable AI Methods")

st.write("")

st.write("""
Choosing the right Explainable AI (XAI) method is critical for research transparency and trust. Our tool is designed to simplify this complex decision process, providing a structured and data-driven approach.

## Key Features:

### 1. Comprehensive Filtering: 

Easily navigate the vast landscape of XAI methods. Our tool allows you to efficiently filter available techniques based on these aspects:

* **Interpretability Scope:** Choose between **Local** (explaining individual predictions) vs. **Global** (explaining overall model behavior).
* **Model Dependence:** Filter for **Model-Agnostic** (works with any model) or **Model-Specific** techniques.
* **Data Type Support:** Ensure the method handles your data (e.g., tabular, images, text).
* **Output Format:** Filter by **Computational Complexity**, required **Fidelity**, and more.

### 2. Decision Making: 

At the core of our tool is the **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** method. TOPSIS is a renowned **Multi-Criteria Decision Making (MCDM)** technique that ranks XAI alternatives by measuring their distance from an ideal best solution and the farthest distance from a negative ideal solution.

### 3. Customizable Criteria and Weights: 

Define your own selection criteria (can be objective or subjective) and assign **objective or subjective weights** to reflect their importance in your project. This ensures the selection process is perfectly tailored to your priorities.

### 4. Best Method Recommendation: 
The application processes your input weights and criteria to produce a clear, ranked list, highlighting the single best-suited method for your particular task context.

"""
)