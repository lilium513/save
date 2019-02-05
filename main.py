import CalcDescSim,CalcTitleSim,CalcCatSim,CalcPermSim
import MeCab
import tools
import  numpy as np
import argparse
from votes import  vote_to_word,vote_to_sentence
from gensim.models import Word2Vec
import pickle
from gensim import corpora
from gensim import models
import gensim
#先行研究→似ているTopKを選出→その中から majority voting で選出(ここで初めて意味的情報を使う)
#問題点TopKを選出する時点で意味論的情報が使われず，良い感じのが捨てられる可能性がある
#特に，この方法で
#評価指標 1.Jaccard係数   2.wordEmbedding間のCos類似度

DESC = 3
TITLE = 2
PERMS = 4
CATEGOLY = 1

WAKATI = 4

LOCATION = 0
MIKE = 1
CAMERA = 2
LMC = [LOCATION,MIKE,CAMERA]

SIMILAR_APPS_NUM = 100
ANSWER_NUMS = 5

TARGET_APP_NUMS = 100

BASE_DIR = "C:/Users/limin/"
PERMISSION_DIR = BASE_DIR +"permission"
ENGLISH_DIR =BASE_DIR + "englishdata"
JAPANESE_DIR =BASE_DIR +  "japanesedata"
CORRESPONDING_DATA = "eandj.txt"
ANSWER_DIR = "answers"
TEXT = "/*.txt"
TSV = "/*.tsv"

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-app", "-application",help="select target app name", default="none")
    # #指定する → その一つについて計算が行われる
    # #指定しない → ある一覧に指定された全てのAPPに対して行われる．
    # parser.add_argument("-c1", "--case1", help="increase output verbosity1", action="store_true")
    # parser.add_argument("-c2", "--case2", help="increase output verbosity2", action="store_true")
    # parser.add_argument("-c3", "--case3", help="increase output verbosity3", action="store_true")

    # コマンドラインから指定があったらそれをターゲットに
    m = MeCab.Tagger("-Ochasen")

    # args = parser.parse_args()
    # target_app_id = args.app
    target_app_id = "a1byte.co.kuks59"
    #コマンドラインから指定がなかったらアプリ一覧のファイルを開いてそこのアプリ全体に対して計算を行う
    if target_app_id  == "none":
        with open(BASE_DIR + "targetapps.txt") as temp:
            target_apps= temp.read()
            target_apps = target_apps.split("\n")
            #TODO loop all apps
            #とりあえずターゲット1個に対して行う
            target_app_id = target_apps[0]
            # appfile→ id \t タイトル \t ジャンル \t 説明文



    with open(ENGLISH_DIR+ "/"+target_app_id + ".tsv",encoding="utf-8") as temp :
        target_app =temp.read()
        target_app = target_app .split("\t")
        target_id = target_app[0]
        target_title = target_app[1]
        target_categoly = target_app[2]
        target_description = target_app[3]
        #appfile→ id \t タイトル \t categoly \t 説明文
    target_app =[""] * 6
    target_app[0] =target_id
    target_app[1] =target_title
    cd  = tools.make_corres_dict()
    target_app[2] =cd[target_categoly]
    target_app[3] =target_description
    target_app[4] = tools.parse_jp(target_description,m)


    w2_model = gensim.models.Word2Vec.load('ja/ja.bin') #TODO アプリのデータも用いて追加学習 load

    # jp_apps,en_apps= tools.load_apps(m)
    # with open("en_apps_dict.pickle", "wb") as f:
    #     pickle.dump(en_apps,f)
    # with open("jp_apps_dict.pickle", "wb") as f:
    #     pickle.dump(jp_apps,f)

    with open("en_apps_dict.pickle","rb") as f:
        en_apps =  pickle.load(f)
    with open("jp_apps_dict.pickle","rb") as f:
        jp_apps=  pickle.load(f)

    with open("permission.pickle","rb") as f:
        p =  pickle.load(f)

    #permissionだけどうこうしたい場合の処理
        # p = tools.load_permission()
    #permissionだけどうこうしたい場合の処理

    for k,v in p.items():
        key = k.split("\\")[1][:-4]
        if key  in en_apps:
            en_apps[key][PERMS] = list(v.values())
        if key in jp_apps:
            jp_apps[key][PERMS] = list(v.values())
    target_app[5] = [0,1,0] #TODO うまい具合に


    answer_apps = tools.load_answer()
    pseudo_jp_apps =[]

    #apps → [path,cat,titile,desc,permidict]
    #desc → 形態素解析済み
    #categoly → 数値で(対応表を使用)
    #for option1 in [True,False]:
    # for option2 in [True,False]:
    # for option3 in [True,False]:
    #jaccards =[]
    #w2vscores = []

    # target_apps = TODO 複数用意しておいて云々江陵
    descriptions = []  # descriptions→[[分かち書きの単語たち],[分かち書きの単語たち],[分かち書きの単語たち],[分かち書きの単語たち]]
    for temp in list(jp_apps.values()):
        if len(temp) != 6:
            print(temp)
        descriptions.append(temp[5])
    descriptions_eng = []  # descriptions→[[分かち書きの単語たち],[分かち書きの単語たち],[分かち書きの単語たち],[分かち書きの単語たち]]
    for temp in list(en_apps.values()):
        if len(temp) != 6:
            print(temp)
        descriptions_eng.append(temp[3])
    # descriptions = list(map (lambda x:m.parse(x[3]).split("\n")[:-2],jp_apps))
    # descriptions = list(map (lambda x:x.split("\t"),descriptions)) #ここ，品詞情報とか残ってる
    # descriptions =
    # descriptions→ [[りんご,食べる],[ラーメン,アイス],[チキン南蛮,スープ],[金沢,寿司]]
    idf = tools.make_idf(descriptions) # idf[word] => value
    jap_cats = list(map (lambda x:x[1],jp_apps.values())) #[1,2,4,11,2,3,4,5,17,9,1]
    en_cats = list(map (lambda x:x[1],en_apps.values())) #[1,2,4,11,2,3,4,5,17,9,1]

    cat_distance = CalcCatSim.calc_cat_sim(descriptions,jap_cats) # cat_distance[x][y] カテゴリxとyとの距離
    cat_distance_eng = CalcCatSim.calc_cat_sim(descriptions_eng,en_cats)

    # phase1:相互情報量を基に権限に対応する単語の決定(共通項目)

    perm_corres_words_noextended = []
    for p in LMC: #pesudo を 使わないパターン TODO ここでpseudo を使うパターン + 特徴語をセーブ
        mi_words = tools.calc_mi(jp_apps, p)
        mi_words = list(map(lambda x:x[1],mi_words))
        perm_corres_words_noextended.append(mi_words)

    # perm_corres_words_extended = []
    # for p in LMC: #pesudo を 使うパターン
    #     perm_corres_words_extended.append(tools.calc_mi(jp_apps+pseudo_jp_apps, p))


    #phase1: 類似アプリの探索()


    scores_old = []
    scores_suggest = []

    avgsl = sum(list(map(lambda x:len(x),descriptions)))/len(descriptions) #記述の平均長さ

    m = MeCab.Tagger("-Ochasen")


    for i,app in enumerate(jp_apps.values()):
        old,suggest = phase1_search_sim_app(target_app,app,idf,avgsl,w2_model,cat_distance,m) #2種類のやり方でターゲットとの距離を計算
        scores_old.append(old) #ターゲットとの類似度を旧手法で計算
        scores_suggest.append(suggest) #ターゲットとの類似度を旧手法で計算
        if i%20 == 0:
            print(str(i) + "/" + str(len(jp_apps)) )

    #計算結果に基づき類似APPベストSIMILAR_APPS_NUM を決定
    with open("sro.pickle","rb") as f:
        scores_old = pickle.load(f)
    with open("srs.pickle","rb") as f:
        scores_suggest = pickle.load(f)

    score_rank_old = np.argsort(scores_old) #as([1,3,2])→[0,2,1]
    score_rank = score_rank_old[::-1]
    apps = list(jp_apps.values())
    apps_old_top_n = []
    for i in score_rank[:SIMILAR_APPS_NUM]:
        apps_old_top_n.append( apps[i])

    #提案手法で類似APPを検索
    score_rank_suggest = np.argsort(scores_suggest)  # as([1,3,2])→[0,2,1]
    score_rank = score_rank_suggest[::-1]
    apps = list(jp_apps.values())
    apps_suggest_top_n = []
    for i in score_rank[:SIMILAR_APPS_NUM]:
        temp = apps[i]
        apps_suggest_top_n.append(apps[i])



    # phase2: 類似アプリから動詞句を抽出 like CLAP
    candidates = []
    for app_top_n in apps_suggest_top_n:
        candidates.extend(app_top_n[DESC].split("。"))
        # TODO . ? でも分割したり良い感じに候補を作ったり
        # windows 上じゃできないっぽいのでどうしたものか
        #とりあえず 。 で分割して候補を作成
    #candidates → [カメラで撮影,りんごを食べる,画像をアップロード,ロシアへ逃亡]



    #phase3 :候補となった群から答えを選択
    candidates_corr_to_perm = [0] * len(LMC)
    for permission in LMC: #権限ごとに文を決定
        candidates_scores = []
        #TODO 拡張データを使った対応語の使用
        perms_words = perm_corres_words_noextended[permission]
        vtw = vote_to_word(candidates,idf,perms_words,m) #単語へのスコアリング辞書
        for candidate in candidates:
            candidates_scores.append(vote_to_sentence(candidate,vtw)) #その辞書を使い文に投票する


        candidates_score_rank = np.argsort(candidates_scores)  # as([1,3,2])→[0,2,1]
        candidates_score_rank = candidates_score_rank[::-1] #[0,2,1]→ [1,2,0]

        candidates = np.array(candidates)
        candidates_top_N = candidates[candidates_score_rank[:ANSWER_NUMS]]

        answers = candidates_top_N
        candidates_corr_to_perm[permission] = answers  #candidatesからスコア上位candidates_top_Nこ


    #phase4 評価
    gold_anmswer = "このアプリはカメラ機能で写真をとってシェアするアプリです"
        #  TODO answer_apps[target_app_id]

    jaccard_score = 0
    w2v_score = 0
    answers = list(map(lambda x:tools.parse_jp(x,m),answers))
    gold_anmswer = tools.parse_jp(gold_anmswer,m)
    for answer in answers:
        jaccard_score += tools.calc_jaccard_distance(answer,gold_anmswer)
        w2v_score += tools.calc_w2v_score(answer,gold_anmswer,w2_model)
    size = len(candidates_top_N)
    jaccard_score /= size
    w2v_score /= size
    print(jaccard_score)
    print(w2v_score)
    #結果をスタック，各アプリケーションについて行う

    #          m1,m2,m3,m4
    # 先行研究
    # 1
    # 2
    # 3
    # 1+2
    # 1+3
    # 2+3
    # 1+2+3

def phase1_search_sim_app(targetApp,app,idf,avgsl,w2_model,cat_distance,m,l1=0.4,l2=0.4,l3=0.1,l4=0.1,l1s=0.4/1000,l2s=0.4/1000):
#cat_distance → cat_distance[cat1][cat2]で カテゴリ間の距離が返ってくる辞書

    s1_old = CalcDescSim.calc_desc_sim(targetApp[WAKATI],app[WAKATI+1],idf,avgsl)
    s1_suggest = CalcDescSim.calc_desc_sim_suggest(targetApp[WAKATI],app[WAKATI+1],idf,avgsl,w2_model)
    s2 = CalcTitleSim.calc_title_sim(targetApp[TITLE-1],app[TITLE],w2_model,m)
    s2_suggest = CalcTitleSim.calc_title_sim_suggested(targetApp[TITLE-1], app[TITLE],idf,avgsl,w2_model,m)
    s3 = cat_distance[targetApp[CATEGOLY+1]][app[CATEGOLY]]
    s4 = CalcPermSim.calc_perm_sim(targetApp[PERMS+1], app[PERMS])
    s_old =  np.dot([s1_old ,s2,s3,s4],[l1,l2,l3,l4])
    s_suggest = np.dot([s1_suggest, s2_suggest, s3, s4], [l1s, l2s, l3, l4])
    return s_old,s_suggest #提案手法とベースラインを同時に計算する．



def getSimilarDescs(descs,target_desc,N=5,lams = [0.5]*4): #target_descの権限にふさわしいテキストをベストNを抽出する
    sim = CalcDescSim.calc_desc_sim()

#
# #全文は新しいアプリの説明文には冗長→部分列も使う
# def makeCandidates(s):
#     S = [s]
#     T = tree(s)
#     for n in T:
#         if n.品詞 == 動詞句:
#             S.append(n.text)
#         if n.品詞 == 接続詞:
#             for n0 in n.parent.children and n0.品詞 != 接続詞:
#                 S.append(n0.text)
#
#     return S
#
# def decidesentences(app,sentences,N):
# #first vote でTop 100くらいを選出，その後それらに対してだけ拡張を実施ある程度見込みがあるやつについて拡張
# #whyperは遅く,これは意味的なマッチングを行えてないので折衷が必要
#
#     return candidates[:N]
#
#
# def textRank(document):
#
#     return [(word,weight),(word,weight),(word,weight),(word,weight)]

if __name__ == "__main__":

    main()