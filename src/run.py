import pandas as pd
from pathlib import Path

from features import engineer_features
from model import *

# Load data
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR /  "test.csv"
SUBMISSION_PATH = DATA_DIR / "submission.csv"

train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)
test_passenger_ids = test_raw['PassengerId']

# Create features
y_train = train_raw['Survived']
X_train = train_raw.drop(columns=['Survived','PassengerId'])
X_test = test_raw.copy()

X_train = engineer_features(X_train)
X_test = engineer_features(X_test)

X_train = pd.get_dummies(X_train, columns=['Embarked', 'Deck'], drop_first=True)
X_test = pd.get_dummies(X_test, columns=['Embarked', 'Deck'], drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Train model
voting_ensemble.fit(X_train, y_train)
predictions = voting_ensemble.predict(X_test)

# Submission file
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': predictions
})

submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Done! '{SUBMISSION_PATH}'")
