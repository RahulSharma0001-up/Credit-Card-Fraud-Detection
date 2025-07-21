# Credit-Card-Fraud-Detection
Credit Card Fraud Detection Using Machine Learning: A Comparative Study
Abstract
Credit card fraud continues to threaten financial institutions and consumers alike, resulting in billions of dollars in losses globally every year. The rise of electronic fraud necessitates the advancement of intelligent and automated detection technologies. This work presents a machine learning-based approach for detecting fraudulent credit card transactions. Various supervised algorithms—Logistic Regression, Support Vector Machine (SVM), Decision Tree, Random Forest, and XGBoost—are studied using a real-world imbalanced dataset. Class balancing, feature scaling, and robust metric evaluation are incorporated. Results indicate that ensemble methods, especially XGBoost and Random Forest, consistently outperform traditional classifiers in terms of precision, recall, F1-score, and AUC-ROC. The findings reinforce the utility of machine learning for real-time, scalable fraud detection systems.
Keywords: Credit Card Fraud, Machine Learning, Imbalanced Data, SMOTE, Ensemble Learning, XGBoost
I. Introduction
With the growth of digital payments, the financial sector faces an ever-evolving risk from credit card fraud. The Nilson Report estimated that worldwide card fraud losses topped $28 billion in 2020. Traditional rule-based systems often fall short against changing fraud patterns and high transaction volumes. Machine learning (ML) presents a dynamic, data-driven approach for real-time anomaly detection.
This study compares the performance of five supervised ML algorithms on a real, highly-imbalanced fraud dataset, with special attention to preprocessing and robust evaluation metrics.
II. Literature Review
•	Dal Pozzolo et al. showcased cost-sensitive learning techniques and outlined the significance of careful feature and model selection in fraud detection.
•	Bhattacharyya et al. highlighted the limitations of accuracy metrics on imbalanced data and called for the use of ROC-based metrics .
•	Jurgovsky et al. demonstrated that sequence-based models (RNNs) could leverage transaction time series for better predictions.
•	Carcillo et al. proposed hybrid approaches combining supervised and unsupervised techniques for performance enhancement.
Building on these works, this paper benchmarks established ML classifiers on a realistic fraud detection task.
III. Dataset Description
The analysis utilizes the Kaggle dataset provided by the Machine Learning Group (ULB) and Worldline, consisting of anonymized European credit card transactions:
•	Total transactions: 284,807
•	Fraudulent transactions: 492 (0.172%)
•	Features: 30 total
o	Time, Amount: Raw features
o	V1 to V28: Principal Component Analysis (PCA) transformed features
o	Class: Target label (1 = fraud, 0 = legitimate)
The extreme class imbalance emphasizes the need for specialized evaluation.
IV. Methodology
A. Data Preprocessing
•	Checked for missing values and outliers; none found.
•	StandardScaler applied to ‘Time’ and ‘Amount’; PCA features were pre-normalized.
•	Stratified 80/20 train-test split to maintain class ratios.
B. Handling Class Imbalance
•	Applied Synthetic Minority Over-sampling Technique (SMOTE) to augment the minority (fraud) class in the training set.
C. Model Selection & Tuning
•	Five algorithms benchmarked: Logistic Regression, SVM, Decision Tree, Random Forest, and XGBoost.
•	Hyperparameters tuned using GridSearchCV and cross-validation (5 folds).
D. Evaluation Metrics
Given the class imbalance, accuracy is not meaningful. Models were compared using:
•	Precision
•	Recall
•	F1-score
•	AUC-ROC
•	Confusion Matrix
V. Results and Discussion
Model	Precision	Recall	F1 Score	AUC-ROC
Logistic Regression	0.89	0.62	0.73	0.95
SVM	0.91	0.63	0.74	0.96
Decision Tree	0.88	0.78	0.83	0.94
Random Forest	0.93	0.85	0.89	0.98
XGBoost	0.95	0.89	0.92	0.99
Feature importance analysis via XGBoost and Random Forest revealed that principal components V4, V14, V12, and ‘Amount’ were among the most significant predictors.
Figures
Figure 1: ROC Curve 
This figure presents:
•	Receiver Operating Characteristic (ROC) curve, illustrating the trade-off between true positive rate and false positive rate; the XGBoost model achieved an AUC of approximately 0.99.
 
VI. Conclusion
This study demonstrates that advanced machine learning—especially ensemble techniques like Random Forest and XGBoost—can dramatically improve credit card fraud detection. Proper data preprocessing and specialized resampling (SMOTE) are essential for handling real-world imbalanced datasets. Future work could extend to sequence modeling (e.g., LSTMs) for temporal transaction data and deploying explainable AI to improve model transparency for bank compliance.
References
1.	The Nilson Report, "Global Card Fraud Losses," Issue 1164, 2021.
2.	A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, “Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy,” IEEE Trans. Neural Netw. Learn. Syst., vol. 29, no. 8, pp. 3784–3797, 2018.
3.	S. Bhattacharyya, S. Jha, K. Tharakunnel, and J. C. Westland, “Data mining for credit card fraud: A comparative study,” Decision Support Systems, vol. 50, no. 3, pp. 602–613, Feb. 2011.
4.	J. Jurgovsky et al., “Sequence classification for credit-card fraud detection,” Expert Syst. Appl., vol. 100, pp. 234–245, 2018.
5.	F. Carcillo, Y.-A. Le Borgne, O. Caelen, and G. Bontempi, “Combining unsupervised and supervised learning in credit card fraud detection,” Information Sciences, vol. 557, pp. 317–331, Mar. 2021.
6.	Kaggle, “Credit Card Fraud Detection,” [Online]. Available: kaggle.com/mlg-ulb/creditcardfraud.
7.	N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic Minority Over-sampling Technique,” Journal of Artificial Intelligence Research, vol. 16, pp. 321–357, 2002.
Appendix
•	Programming Environment: Python 3.9, scikit-learn, XGBoost, imbalanced-learn, Pandas, Matplotlib.
•	Notebook and Source Code: 

