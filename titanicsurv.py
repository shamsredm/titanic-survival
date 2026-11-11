# Titanic survival prediction
# Final model: random forest (max depth 8) with Pclass, Sex, Age, Fare,
# FamilySize and Title features. Prints the accuracy on a held-out test set.
# The cross-validation loop below is commented out; uncomment it to
# reproduce the CV scores in the README.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

df = pd.read_csv("titanic.csv")

# 1. Cleaning the data: fill missing ages and turn Sex into 0/1
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# 2. Build extra clues to help accuracy: Title and FamilySize
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
common = ["Mr", "Miss", "Mrs", "Master"]
df["Title"] = df["Title"].where(df["Title"].isin(common), "Rare")
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 3. Turning Title into 0/1 columns
df = pd.get_dummies(df, columns=["Title"], dtype=int)
title_cols = [c for c in df.columns if c.startswith("Title_")]

# 4. Picking the clues
features = df[["Pclass", "Sex", "Age", "Fare", "FamilySize"] + title_cols]
target = df["Survived"]

# 5. Cross-validation at two depths (commented out; results are in the README)
# for depth in [5, 8]:
#     model = RandomForestClassifier(max_depth=depth, random_state=42)
#     score = cross_val_score(model, features, target, cv=5).mean()
#     print("depth", depth, "CV accuracy:", round(score, 3))

# 6. Final score on a held-out test set
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=0
)
final = RandomForestClassifier(max_depth=8, random_state=42)
final.fit(X_train, y_train)
print("final test accuracy:", final.score(X_test, y_test))
