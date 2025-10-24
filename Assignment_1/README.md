# Assignment 1 – Supervised Learning Foundations (KNN & Regression Concepts)

**Goal:**  
Formulate a predictive learning problem from scratch and connect the abstract definitions  
(sample space $\mathcal{X}$, label space $\mathcal{Y}$, loss $\ell$, distance $d$, and evaluation metrics)  
to a practical example — predicting students’ final *Machine Learning A* grades.

---

## Problem Definition
We imagine a model predicting the final grade of a student based on prior academic data:

| Variable | Description | Type |
|-----------|--------------|------|
| $X_1$ | Linear Algebra grade | continuous [0–100] |
| $X_2$ | Probability & Statistics grade | continuous [0–100] |
| $X_3$ | Introduction to Programming grade | continuous [0–100] |
| $X_4$ | Completed Python course (0 = No, 1 = Yes) | binary |
| $X_5$ | Department (1 = Math, 2 = Statistics, 3 = CS) | categorical |

Hence  

- Sample space  $ \mathcal{X} \subset \mathbb{R}^5 \times \{0,1,2,3\}$  
- Label space  $ \mathcal{Y} = [0,100] \subset \mathbb{R}$ represents final grades.

---

## Methodology

1. **Loss Function**  
   Mean Squared Error (MSE):  

   $$
   \text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
   $$

2. **Algorithm**  
   Implemented the *K-Nearest Neighbors* (KNN) regressor using Euclidean distance:

   $$
   d(x, x') = \sqrt{\sum_{j=1}^{d}(x_j - x'_j)^2}
   $$

3. **Validation Strategy**  
   Split the dataset into **train / test** subsets to estimate predictive performance.  
   Computed **MSE** and **accuracy** metrics.  

4. **Bias & Overfitting Discussion**  
   Illustrated how small $K \Rightarrow$ high variance, large $K \Rightarrow$ high bias,  
   and how limited data causes overfitting.

---

##  Insights
- Formalized each component of a supervised-learning pipeline ($\mathcal{X}, \mathcal{Y}, \ell, d$).  
- Connected theoretical definitions to a realistic prediction problem.  
- Discussed validation bias and the role of the i.i.d. assumption.  
- Built intuition for later topics such as empirical-risk minimization and generalization bounds.

---

## Skills Demonstrated
Mathematical Modeling · KNN · Validation Design · Bias–Variance Analysis · Python (NumPy / Matplotlib)

---

## Demo Notebook
See the accompanying Jupyter Notebook [`knn_distance_demo.ipynb`](knn_distance_demo.ipynb)  
for a short visual demonstration of KNN regression on synthetic data.

---

_This project is an independent reconstruction of my learning outcomes from  
**Machine Learning A** at the University of Copenhagen.  
All code and text are original and intended solely for portfolio and educational purposes._
