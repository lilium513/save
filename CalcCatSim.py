# 計算方法→tf-idfでカテゴリベクトルを作って，そのcos類で
from tools import make_tf_idf_ndl, calc_cos_sim ,make_idf,calc_tf
from collections import defaultdict
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def calc_cat_sim(docs,categolies):

    cat_sim = defaultdict(lambda: defaultdict(lambda: 0) )  # cat_sim[A][B]→カテゴリAとBの類似度
    size = max(categolies)
    docs_classifieds = [""] *(size + 1)  #size個の文章群
    for cat,doc in zip(categolies,docs):
        docs_classifieds[cat]+= " ".join(doc)

    vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
    vecs = vectorizer.fit_transform(docs_classifieds)
    tf = vecs.toarray()

    # with open('cat_tf_idf.pickle', 'rb') as f:
    # tf =obj = pickle.load(f)
    for i, c1 in enumerate(tf):
        for j, c2 in enumerate(tf):

            cat_sim[i][j] = calc_cos_sim(c1, c2)

    return cat_sim  # invalid value encountered in double_scalars[i][j]→カテゴリiとカテゴリjの間の類似度
