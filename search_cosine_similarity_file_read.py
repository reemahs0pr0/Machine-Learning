import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#%%
file = open('../data/quotes.txt', encoding='utf-8')
doc = file.read()
file.close()

new_doc = ""

for i in range(12,len(doc)-3):
    new_doc += doc[i]

split = new_doc.split("\"")

corpus = []
for i in range(121):
    if i % 2 ==1:
        corpus.append(split[i])
#%%
stop_words = stopwords.words('english') 

porter = nltk.stem.PorterStemmer()

docs = []
punc = str.maketrans('','', string.punctuation)
for doc in corpus:
    doc_no_punc = doc.translate(punc)
    words_stemmed = [porter.stem(w) for w in doc_no_punc.lower().split()
    	if not w in stop_words]
    docs += [' '.join(words_stemmed)]

print(docs)

tfidf_vec = TfidfVectorizer()
tfidf_vec.fit(docs)
tfidf_wm = tfidf_vec.transform(docs).toarray()

features = tfidf_vec.get_feature_names()
indexes = ['doc'+str(i) for i in range(len(corpus))]
tfidf_df = pd.DataFrame(data=tfidf_wm, index=indexes, columns=features)
print(tfidf_df)

query = 'life wise choices'
query = query.translate(punc)	# remove punctuation
query_arr = [' '.join([porter.stem(w) for w in query.lower().split()])]

tfidf_wm2 = tfidf_vec.transform(query_arr).toarray()
print(tfidf_wm2)

print("")
docs_similarity = cosine_similarity(tfidf_wm2, tfidf_wm)
query_similarity = docs_similarity[0]

series = pd.Series(query_similarity, index=tfidf_df.index)
sorted_series = series.sort_values(ascending=False)
sorted_series = sorted_series[sorted_series!=0]
print(sorted_series)

print("\nSearch results for query: '", query, "':\n", sep='')

for index in sorted_series.index:
	doc_idx = int(index[3:])
	print(corpus[doc_idx], " [score = ", sorted_series[index], "]\n", sep='')

