import pandas as pd
import pickle
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Loading data...")
df = pd.read_csv('data/train.csv')
df['Review'] = df['Review'].fillna('')
y = (df['Rating'] > 3).astype(int)

print("Training vectorizer...")
cv = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X = cv.fit_transform(df['Review'])

print("Training model...")
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X, y)

print("Saving models...")
with open('model.pickle', 'wb') as f:
    pickle.dump(cv, f)
joblib.dump(model, 'model.sav')

print("Done!")
