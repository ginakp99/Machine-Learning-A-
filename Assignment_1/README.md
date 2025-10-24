# Assignment 1 – Supervised Learning Foundations (KNN & Regression Concepts)

**Goal:**  
Formulate a predictive learning problem from scratch and connect the abstract definitions  
(sample space 𝒳, label space 𝒴, loss ℓ, distance d, evaluation)  
to a practical example: predicting students’ final Machine Learning grade.

---

## Problem Definition
We imagine a model predicting the final grade of a student based on prior academic data:

| Variable | Description | Type |
|-----------|--------------|------|
| X₁ | Linear Algebra grade | continuous [0–100] |
| X₂ | Probability & Statistics grade | continuous [0–100] |
| X₃ | Introduction to Programming grade | continuous [0–100] |
| X₄ | Completed Python course (0 = No, 1 = Yes) | binary |
| X₅ | Department (1 = Math, 2 = Statistics, 3 = CS) | categorical |

Thus 𝒳 ⊂ ℝ⁵ × {0, 1, 2, 3}, and  
the label space 𝒴 = [0, 100] ⊂ ℝ represents final grades.

---

## Methodology
1. **Loss function:**  
   Mean Squared Error  
   \[
   \text{MSE}=\frac1n\sum_i (y_i-\hat y_i)^2
   \]
2. **Algorithm:**  
   Implemented the *K-Nearest Neighbors* regressor using **Euclidean distance**  
   \[
   d(x,x')=\sqrt{\sum_j (x_j-x'_j)^2}
   \]
3. **Validation strategy:**  
   Split data into **train / test** sets to estimate predictive performance.  
   Computed MSE and accuracy metrics.
4. **Bias & overfitting discussion:**  
   Illustrated how small K → high variance, large K → high bias,  
   and how limited data causes overfitting.

---

##  Insights
- Formalized every component of a supervised-learning pipeline.  
- Connected theoretical definitions to an intuitive real-world task.  
- Discussed validation bias and the i.i.d. assumption.  
- Built foundation for later assignments on empirical-risk minimization and generalization bounds.

---

##  Skills Demonstrated
Mathematical Modelling · KNN · Validation Design · Bias–Variance Analysis · Python (Numpy/Matplotlib)

---

_This project is an independent reconstruction of my learning outcomes from Machine Learning A (UCPH),  
intended for portfolio and educational purposes._
