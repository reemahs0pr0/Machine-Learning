# corpus = [
#     "life is the sum of our choices",
#     "you attract into your life that which you are",
#     "don't be afraid your life will end be afraid that it will never begin"
#     ]

corpus = [
    'John has some cats',
    'cats eat fish',
    'I eat a big fish'
    ]

# corpus = ['eat fish']

docs = []
tokens = []

for sentence in corpus:
    docs.append(sentence.split(" "))
    tokens.extend(sentence.split(" "))
    
tokens = set(tokens)
tokens = list(tokens)

stop_words = ["has", "some", "a", "I"]
for stop_word in stop_words:
    if stop_word in tokens:
        tokens.remove(stop_word)
    for doc in docs:
        for word in doc:
            if stop_word == word:
                doc.remove(word)

bow_dict = {}
for i in range(len(docs)):
    for token in tokens:
        if token not in bow_dict:
            bow_dict[token] = []
        count = 0
        for word in docs[i]:
            if token == word:
                count += 1
        bow_dict[token].append(("doc" + str(i+1), count))

import math

idf_dict = {}
for token in bow_dict:
    freq = 0
    for tup in bow_dict[token]:
        freq += tup[1]
    idf_dict[token] = math.log((1+len(docs))/(1+freq), math.e)+1

tf_dict = {}
for i in range(len(docs)):
    for token in tokens:
        if token not in tf_dict:
            tf_dict[token] = []
        count = bow_dict[token][i][1]
        tf_dict[token].append(("doc" + str(i+1), count/len(docs[i])))
    
unnorm_tfidf_dict = {}
for i in range(len(docs)):
    for token in tokens:
        if token not in unnorm_tfidf_dict:
            unnorm_tfidf_dict[token] = []
        tf = tf_dict[token][i][1]
        unnorm_tfidf_dict[token].append(("doc" + str(i+1), tf*idf_dict[token]))

magnitude = []    
for i in range(len(docs)):
    total = 0
    for token in tokens:
        total += pow(unnorm_tfidf_dict[token][i][1], 2)
    magnitude.append(pow(total, 0.5))
        
norm_tfidf_dict = {}
for i in range(len(docs)):
    for token in tokens:
        if token not in norm_tfidf_dict:
            norm_tfidf_dict[token] = []
        tfidf = unnorm_tfidf_dict[token][i][1]
        norm_tfidf_dict[token].append(("doc" + str(i+1), tfidf/magnitude[i]))