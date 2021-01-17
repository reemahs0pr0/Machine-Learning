query = {
    'eat': 0.707107,
    'fish': 0.707107
    }
doc = {
       'eat': 0.57735,
       'fish': 0.57735,
       'cat': 0.57735
       }

def dot(dict_1, dict_2):
    total = 0
    for key in dict_1:
        if key in dict_2:
            total += dict_1[key] * dict_2[key]
    return total

def length(dictionary):
    total = 0
    for key in dictionary:
        total += pow(dictionary[key], 2)
    return pow(total, 0.5)

def cosine_similarity(dict_1, dict_2):
    return dot(dict_1, dict_2)/(length(dict_1)*length(dict_2))

print(cosine_similarity(query, doc))