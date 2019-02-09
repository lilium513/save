# 各単語をw2vでエンベッド→各単語の類似度の平均をとる
# title1 "w1 w2 w3" title2 "w4 w5 w6" → 14 15 16 24 25 26 34 35 36の平均をとる
# 0.4以下は0に
import MeCab
from tools import  calc_cos_sim,original_text_to_parsed
from CalcDescSim import calc_desc_sim_suggest
import tools

def calc_title_sim(t1, t2,w2model =None ,m = None): #t1,t2→生のタイトル 「Yahoo乗り換え案内」

    t1_words = tools.parse_jp(t1,m)
    t2_words = tools.parse_jp(t2,m)
    nums =len(t1_words) * len(t2_words)
    if nums == 0:
        return 0
    sim_sum = 0
    for t1_word in t1_words:
        for t2_word in t2_words:
            #vec → w2v ベクトル化
            if t1_word not in w2model.wv or t2_word not in w2model.wv:
                pass
            else:
                wv1 = w2model.wv[t1_word]
                wv2 = w2model.wv[t2_word]
                cs = calc_cos_sim(wv1,wv2)
                if cs >= 0.4:
                    sim_sum += cs

    return sim_sum/nums

def calc_title_sim_suggested(t1_words, t2_words,idf,avgsl,w2model,m,normdic,w2dot,wtoind,cossim):

    return calc_desc_sim_suggest(t1_words,t2_words,idf,avgsl,w2model,normdic,w2dot,wtoind,cossim)