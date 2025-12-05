import streamlit as st
import subprocess

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
* **Output Format:** Select from **Visual**, **Numerical**, **Textual**, and **Rules**.

### 2. Decision Making: 

The tool automatically processes the decision matrix and evaluates it using a set of **10 MCDM techniques** to ensure the quality of the solution. **Borda Count** provides a balanced overall ranking of XAI methods.

### 3. Customizable Criteria and Weights: 

Define your own selection criteria (can be objective or subjective) and assign **objective or subjective weights** to reflect their importance in your project. This ensures the selection process is perfectly tailored to your priorities.

### 4. Best Method Recommendation: 
The application processes your input weights and criteria to produce a clear, ranked list, highlighting the single best-suited method for your particular task context.
**Sensitivity analysis** observe how the resulting order of methods changes when modifying the weights.

"""

)

try:
    # Použijeme pip freeze, ktorý dáva výstup vo formáte requirements.txt
    result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True, check=True)
    package_list = result.stdout

    st.subheader("Verzie nainštalovaných knižníc (pip freeze):")
    st.code(package_list, language='text')

except subprocess.CalledProcessError as e:
    st.error(f"Nepodarilo sa získať zoznam balíkov: {e}")
except FileNotFoundError:
    st.error("Príkaz 'pip' nebol nájdený.")



