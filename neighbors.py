from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

text = ['testing', 'test', 'this is a test', 'why and I still testing']
def doc_sampler(input_docs, reference_docs):
    c = CountVectorizer(analyzer = 'char')
    ref_x = c.fit_transform(reference_docs)
    input_x = c.transform(input_docs)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(ref_x)
    _, indices = nbrs.kneighbors(input_x)
    index = np.array([v[1] for v in indices])
    return np.array(reference_docs)[index]
print(doc_sampler(text, text))
