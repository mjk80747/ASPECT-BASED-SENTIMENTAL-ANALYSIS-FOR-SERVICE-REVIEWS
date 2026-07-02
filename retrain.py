import pandas as pd
import pickle
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

print("Loading data...")
df = pd.read_csv('data/train.csv').sample(2000, random_state=42)
df['Review'] = df['Review'].fillna('')
y = (df['Rating'] > 3).astype(int)

print("Training vectorizer...")
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(df['Review']).toarray()

print("Training model...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

print("Saving models...")
with open('model.pickle', 'wb') as f:
    pickle.dump(cv, f)
joblib.dump(model, 'model.sav')

print("Done!")
