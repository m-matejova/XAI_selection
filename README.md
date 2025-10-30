# 🤖 A Multi-Criteria Decision Making Approach for the Selection of Explainable AI Methods

[![GitHub license](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📝 Project Overview

This project presents an [Tool](https://xai-selection.streamlit.app/)for selecting the most appropriate Explainable Artificial Intelligence (XAI) method for a given application scenario. Traditional selection of XAI techniques (such as LIME, SHAP, or Grad-CAM) often relies on subjective intuition or anecdotal evidence, which is insufficient for high-stakes domains requiring rigorous validation.

Our system resolves this issue by implementing **Multi-Criteria Decision-Making (MCDM)** techniques, to systematically compare and rank XAI methods based on a set of **subjective and objective metrics**.

---

## ✨ Key Features

* **Different approaches to determining metric weights:**
    * ***Direct rating:*** Allows you to enter weights directly on a scale from 1 to 10.
    * ***Pairwise comparison:*** Comparing pairs of metrics and subjectively determining their importance.
    * ***Objective Criteria Weighting (CRITIC):*** Uses inherent statistical data properties (variability and inter-criteria correlation) to transparently establish importance weights for each evaluation metric.
* **Transparent Outputs:** Generates a final, ranked list of XAI methods, minimizing human subjectivity and maximizing transparency.

---



## 🚀 Usage

The project includes all the necessary materials to reproduce the results mentioned in the study.
