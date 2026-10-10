Titanic Survival Classification (Logistic Regression from Scratch)
A beginner-friendly binary classification project that predicts passenger survival on the Titanic using regularized logistic regression implemented from scratch with NumPy — no scikit-learn for the model itself.

📌 Overview
The goal is to predict whether a passenger survived the Titanic disaster (Survived: 0 = No, 1 = Yes) using features such as passenger class, sex, age, and family relations.

This project demonstrates the full ML workflow:

Data cleaning and preprocessing

Feature encoding and scaling

Manual train/test split

Gradient descent optimization with L2 regularization

Model evaluation on a holdout test set

The model is intentionally written with explicit loops and matrix operations to make the math behind logistic regression transparent.

📂 Project Structure
text
.
├── Titanic-Dataset.csv     # Raw dataset (891 rows, 12 columns)
├── train.csv               # Generated 80% training split
├── test.csv                # Generated 20% test split
├── logistic_regression.py  # Main script
└── README.md
🧾 Dataset
Source: Titanic - Machine Learning from Disaster (Kaggle)

Column	Description
PassengerId	Unique ID (dropped)
Survived	Target: 0 = died, 1 = survived
Pclass	Ticket class (1, 2, 3)
Name	Passenger name (dropped)
Sex	male / female
Age	Age in years
SibSp	# siblings / spouses aboard
Parch	# parents / children aboard
Ticket	Ticket number (dropped)
Fare	Passenger fare (dropped in base model)
Cabin	Cabin number (dropped)
Embarked	Port of embarkation (dropped)
Rows with missing Age are removed to keep the demo simple.

⚙️ Approach
1. Preprocessing
Drop non-informative columns (Ticket, Fare, PassengerId, Name, Cabin, Embarked).

Drop rows with missing Age.

Encode Sex as male = 0, female = 1.

Shuffle the dataset (fixed random_state=42 for reproducibility).

Split 80% train / 20% test.

2. Feature Scaling
Standardization (z-score normalization):

text
x' = (x - μ) / σ
This is critical for gradient descent convergence when features have very different ranges (e.g. Age vs SibSp).

3. Model: Logistic Regression
Hypothesis:

text
z = XW + b
f(x) = σ(z) = 1 / (1 + e^(-z))
4. Cost Function (with L2 Regularization)
text
J(W, b) = -(1/m) Σ [ y·log(f) + (1-y)·log(1-f) ] + (λ / 2m) Σ W²
First term: binary cross-entropy loss

Second term: L2 penalty that discourages large weights

Bias b is not regularized

5. Gradients
text
∂J/∂W = (1/m) · Xᵀ(f - y) + (λ/m) · W
∂J/∂b = (1/m) · Σ(f - y)
6. Parameter Update
text
W ← W - α · ∂J/∂W
b ← b - α · ∂J/∂b
🔧 Hyperparameters
Parameter	Value	Notes
epochs	1000	Full-batch gradient descent iterations
alpha	0.01	Learning rate
reg_lambda	0.1	L2 regularization strength
W init	zeros	Standard for logistic regression
b init	0.0	—
Tune reg_lambda to control overfitting:

λ = 0 → no regularization

λ = 0.01 → light

λ = 0.1 → moderate (default)

λ = 1.0+ → strong

📊 Results
Metric	Value
Test Accuracy	~77%
Sample training log:

text
Epoch  200 | cost = 0.553192
Epoch  300 | cost = 0.521447
Epoch  400 | cost = 0.500817
Epoch  500 | cost = 0.486765
Epoch  600 | cost = 0.476808
Epoch  700 | cost = 0.469517
Epoch  800 | cost = 0.464030
Epoch  900 | cost = 0.459806
Test accuracy: 76.92%
🚀 How to Run
Requirements
bash
pip install numpy pandas
Execute
bash
python logistic_regression.py
Make sure Titanic-Dataset.csv is in the same directory as the script.

🐛 Known Issues / Fixes
Missing bias in test prediction

python
z = np.dot(X[_], W)         # ❌ wrong
z = np.dot(X[_], W) + b     # ✅ correct
Dropping rows with missing Age loses ~20% of data. Better: impute with median grouped by Title and Pclass.

Dropping Fare, Embarked, Cabin, Name discards strong predictors (e.g. Fare correlates strongly with survival).

📈 How to Improve Accuracy
The base model caps around 76–82% on Titanic. To push toward 82–86% (realistic ceiling — 90%+ usually means overfitting or leakage):

Feature Engineering
Extract Title from Name (Mr, Mrs, Miss, Master, Rare)

Create FamilySize = SibSp + Parch + 1

Create IsAlone = (FamilySize == 1)

Extract Deck from Cabin, and HasCabin

Fare_log = log1p(Fare) to reduce skew

Impute Age by (Title, Pclass) group medians

Fill missing Embarked with mode

Model Improvements
Add polynomial / interaction features (Sex × Pclass, Age × Pclass)

Try scikit-learn's LogisticRegression with C tuned via GridSearchCV

Try tree ensembles: Random Forest, Gradient Boosting, XGBoost, LightGBM

Use StackingClassifier or VotingClassifier

Validation
Replace single holdout with Stratified K-Fold Cross-Validation (k=5 or 10)

Use repeated CV for stable estimates

📚 Key Takeaways
Logistic regression is a linear classifier; feature scaling is non-negotiable for gradient descent.

L2 regularization shrinks weights and reduces overfitting — but it cannot compensate for missing features.

On small tabular datasets like Titanic, feature engineering matters far more than model choice.

A model trained from scratch is a great way to internalize the math, but production work should use a battle-tested library.

🧠 References
Andrew Ng — Machine Learning (Coursera)

Kaggle Titanic Competition

scikit-learn LogisticRegression docs

📄 License
This project is for educational purposes. The Titanic dataset is publicly available on Kaggle.