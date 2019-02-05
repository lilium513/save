from sklearn.feature_extraction.text import TfidfVectorizer

# 文書
documents = ['a b c a d',
'c b c',
'b b a',
'a c c',
'c b a']

vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
vecs = vectorizer.fit_transform(documents)

print (vecs.toarray())