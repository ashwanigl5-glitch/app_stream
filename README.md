ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.9012	0.9054	0.644	0.3488	0.4525	0.4264
Decision Tree	0.8776	0.7134	0.4778	0.4991	0.4882	0.4189
kNN	0.8936	0.8084	0.586	0.3091	0.4047	0.3742
Naive Bayes	0.8639	0.8088	0.4282	0.4877	0.456	0.3797
Random Forest (Ensemble)	0.9049	0.9272	0.6581	0.3894	0.4893	0.4592


ML Model Name	Observation about model performance
Logistic Regression	High performance baseline with 90.12% Accuracy and 0.9054 AUC. It shows balanced linear separation but struggles somewhat with identifying minority positive classes (34.88% Recall).
Decision Tree	Prone to overfitting. While it achieves the highest raw Recall (49.91%) among single models, its low AUC (0.7134) and Precision (0.4778) indicate high false-positive rates.
kNN	Decent non-linear distance baseline (89.36% Accuracy), but shows weak minority class capturing capability (30.91% Recall) due to high local data cluster density.
Naive Bayes	Shows the lowest overall structural Accuracy (86.39%) but acts as a strong generalized separator with a competitive Recall of 0.4877 and an AUC of 0.8088.
Random Forest (Ensemble)	Outstanding generalisation capabilities. It leads across almost all key stability criteria: 90.49% Accuracy, 0.9272 AUC, 65.81% Precision, and an MCC of 0.4592.
Overall Winner for your dataset?	Random Forest (Ensemble). It handles the target class imbalance smoothly, outputting the highest discriminative power (AUC) and best overall classification balance (MCC).




