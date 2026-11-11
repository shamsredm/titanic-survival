# Titanic Survival Prediction

**Goal:** Goal: predict whether a passenger survived, from class, sex, age, fare, family size and titles.

## Credit
Kaggle Titanic Dataset
  
## Data and cleaning
- 891 passengers, 12 columns
- Missing: Age (177, filled with the median), Cabin (687, not used), Embarked (2, not used)
- Sex converted to 0/1

## Results

| Model | Score |
|---|---|
| Always guess "died" (baseline) | about 59-62% |
| Logistic regression (5-fold CV) | 78.7% |
| Random forest, no depth limit (CV) | 81.6% |
| Random forest, max depth 8 (CV) | 83.7% |
| + Title and FamilySize features (CV) | 82.7% / 83.2% (no clear gain) |
| **Final held-out test accuracy** | **84.4%** |

*Note: `titanicsurv.py` reproduces the final model and its test accuracy. The CV loop in it is commented out (uncomment it to reproduce the CV scores), and the baseline, logistic regression, and unlimited-depth forest numbers come from earlier experiments that the script doesn't include.*

## What I learned

**Overfitting.** A random forest with no depth limit scored 97.8% on its own
training data but only about 82% on passengers it hadn't seen. With no limit,
the trees keep asking questions until they've memorized individual passengers
instead of learning patterns that apply to many people. Limiting the depth to 8
forced simpler questions and raised the honest (cross-validated) score to 83.7%.

**Why cross-validation.** The same logistic regression model scored anywhere
from 74.9% to 80.4% depending on which 20% of passengers was hidden, so a single
train/test split can be lucky or unlucky. 5-fold cross-validation tests every
passenger once and averages the results. I also kept a separate test set that
no decision touched, and the final model scored 84.4% on it, which was matched 
close to the cross-validation estimate.

**What mattered for survival.** Sex and class dominated. First-class women
survived 96.8% of the time and third-class men only 13.5%, and a third-class
woman (50.0%) was more likely to survive than a first-class man (36.9%). The
title "Master" (young boys) survived at 57.5% against 15.7% for "Mr", even
though both are male. All of this is consistent with a "women and children
first" policy, though the data alone can't prove the cause.

**What didn't help.** Adding the passenger's title and family size did not
improve the model beyond the noise (about one point): CV scores were 82.7% at
depth 5 and 83.2% at depth 8, against 82.2% and 83.7% without them. Family size
did show a pattern: families of 2 to 4 survived more often than solo travellers,
and families of 5 or more did worse. But solo travellers were about 77% men and
the largest families were entirely third class, so the feature mostly repeated
what sex and class already told the model. A new clue only helps if it carries
information the existing columns don't.

## How to run
pip install pandas scikit-learn
python titanicsurv.py
