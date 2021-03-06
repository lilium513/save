# d → [りんご,食べる]
# [写真,撮る]
# calc desc sim
# トピックモデルだと粒度が粗い(SMS と Email)→ので単語レベルでやってる→だが今度は細かすぎる(e.g, 撮影 と 撮る が別視される)
#docs → [[分かち書き説明文1],[分かち書き説明文2],[分かち書き説明文3],...]
import pickle
from collections import defaultdict
from tools import make_vocab,calc_cos_sim,calc_tf
from gensim.models import word2vec
from votes import vote_to_sentence,vote_to_featuring_word
import  tools
#old method
def calc_ocapi_25(doc,idf,avgsl,voc,K1 = 0.5,b = 0.5): #parsed_d → [私,りんご,食べる]
    keys = voc
    values = len(voc) * [0]
    o_25_dic = dict(zip(keys,values))

    ndl = len(doc) / avgsl
    # tf[index][word]
    # idf[word]
    # ndl[index]
    # calc_tf(word,doc)

    tf = calc_tf(doc)
    for word in tf:
        if word in idf:
            o25  = (tf[word] * idf[word] * (K1 + 1)) / (K1 * (1 - b + (b * ndl) + tf[word]))
            o_25_dic[word] = o25

    return list(o_25_dic.values())

def calc_desc_sim(d1,d2,idf,avgsl):
    voc = tools.make_vocab([d1,d2])
    ocapi_25_d1 = calc_ocapi_25(d1,idf,avgsl,voc)
    ocapi_25_d2 = calc_ocapi_25(d2,idf,avgsl,voc)
    return calc_cos_sim(ocapi_25_d1,ocapi_25_d2)

#suggested method
def calc_desc_sim_suggest(sl,ss,idf,avgsl,model,normdic,dot_dic_w2v,wtoind,cossim,k1=0.5,b = 0.5):
    result = 0
    l = len(ss)
    for w in sl:
        if w in idf:
            temp_sem = sem(w,ss,model,normdic,dot_dic_w2v,wtoind,cossim)
            result += idf[w]* (
                    (temp_sem*(k1+1))
                    /
                    (temp_sem+(k1)*(1-b+b*(l/avgsl)))
            )

    return result


def sem(word,doc,w2model,normdic,dot_dic_w2v,wtoind,cossim): #TODO dotに関して動的計画方を実行する
    word_vector = w2model.wv
    if word not in word_vector :
        return 0
    # v1 = w2model.wv[word]
    max_value = 0
    for w2 in doc:
        if w2 not in word_vector :
            continue
        # v2 = w2model.wv[w2]
        # d = tools.calc_cos_sim_using_normdic(word,w2,v1,v2,normdic,dot_dic_w2v,wtoind)
        d = cossim[wtoind[w2]][wtoind[word]]
        if (d) > max_value:
            max_value = d
    return max_value

#hogehoge_suggested → 提案手法を用いた類似度測定
#hogehoge → 提案手法を用いた類似度測定

