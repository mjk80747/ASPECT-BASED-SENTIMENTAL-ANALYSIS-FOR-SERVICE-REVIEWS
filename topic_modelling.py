import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from collections import defaultdict,Counter
from nltk import word_tokenize
import pandas as pd
import numpy as np

_required_nltk = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet',
    'omw-1.4': 'corpora/omw-1.4',
}
for pkg, resource in _required_nltk.items():
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(pkg)

def Topic_modeling(df):

    stop_words=list(nltk.corpus.stopwords.words('english'))
    df['clean_doc'] = df['sentence'].str.replace("[^a-zA-Z#]", " ")

    # cleaning the text
    def clean_text(headline):
        le=WordNetLemmatizer()
        word_tokens=word_tokenize(headline)
        tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words]
        cleaned_text=" ".join(tokens)
        return cleaned_text
    df['clean_doc'] = df['clean_doc'].apply(clean_text)

    # Tf-idf for the data
    vect =TfidfVectorizer(stop_words=stop_words,max_features=1000)
    vect_text=vect.fit_transform(df['clean_doc'])

    # Topic_modeling Algo
    from sklearn.decomposition import LatentDirichletAllocation
    lda_model=LatentDirichletAllocation(n_components=2)
    lda_top=lda_model.fit_transform(vect_text)

    #Top 10 Words that has more impact on the topic:
    vocab = vect.get_feature_names_out()
    topic = defaultdict(list)
    for i, comp in enumerate(lda_model.components_):
        vocab_comp = zip(vocab, comp)
        sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]

        for t in sorted_words:
            topic[i].append(t[0])

    # Most prominent topic for the paragraph
    top = []
    for i in range(len(df)):
        top.append(np.argmax(lda_top[i]))
    c = Counter(top)
    ma,ti = -1,-1
    for i,j in c.items():
        if j>ma:
            ma = j
            ti = i
    return ti,topic[ti]
