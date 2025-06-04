import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load Titanic dataset
df = sns.load_dataset("titanic")

# 2. Preprocess: Drop NA & select relevant features
df = df[["survived", "pclass", "sex", "age", "fare", "embarked"]].dropna()

# Encode categorical vars
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["embarked"] = df["embarked"].map({"S": 0, "C": 1, "Q": 2})

# Features & target
X = df.drop("survived", axis=1)
y = df["survived"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Train Decision Tree
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 5. Visualization
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=["Died", "Survived"], filled=True)
plt.title("Decision Tree - Titanic Survivors")
plt.show()

# 6. Evaluation
y_pred = clf.predict(X_test)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
