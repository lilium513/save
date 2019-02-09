#tools → 共通して使いそうなやつの纏め
import random

import gensim
import pickle

import numpy as np
from collections import defaultdict
import math
import MeCab
import glob
import os
import copy
from googletrans import Translator
import traceback
import re
import collections
BASE_DIR = "C:/Users/limin/"
PERMISSION_DIR = BASE_DIR +"permission"
ENGLISH_DIR =BASE_DIR + "englishdata"
JAPANESE_DIR =BASE_DIR +  "japanesedata"
CORRESPONDING_DATA = "eandj.txt"
ANSWER_DIR = BASE_DIR + "answers"
TEXT = "/*.txt"
TSV = "/*.tsv"
DATA_FILE = "hogehoge.fuga" #いちいちロードするのは億劫なので一度計算したらセーブしてロードするようにする

PER_DIR_LEN = 26
CAT = 1
TITLE = 2
DESC = 3
PERM = 4
WAKATI = 5
POSTAG = 3
GENKEI = 2


def make_vocab(docs): #形態素解析して単語の配列になったものをボキャブラリー化する
    vocab = []
    for doc in docs:
        vocab.extend(doc)
    vocab = list(set(vocab))
    return vocab

def calc_w2v_score(text1,text2,w2vmodel):
    total = 0
    nums = len(text1) * len(text2)
    for w1 in text1:
        for w2 in text2:
            if w1 in w2vmodel.wv and w2 in w2vmodel.wv:
                w1_to_w2v = w2vmodel.wv[w1]
                w2_to_w2v = w2vmodel.wv[w2]
                total += calc_cos_sim(w1_to_w2v,w2_to_w2v)
    if nums ==0:
        return 0
    return total/nums
def calc_jaccard_distance(a,b): #引数 aもbもリスト a = ["けもの","いても","のけもの","いない"]b = ["けもの","フレンズ","パビリオン"]
    a = set(a)
    b = set(b)
    num_inter = len(a & b ) #分子
    num_union = len( a | b ) #分母
    if num_union == 0:
        return 0
    return num_inter/num_union

def load_permission():
    permissions_texts_dirs = glob.glob(PERMISSION_DIR + TEXT)
    dic = {}
    # 権限のみの読み込み
    for dir in permissions_texts_dirs:
        # 権限の処理
        with open(dir, "r", encoding="utf-8") as temp:
            keys = ["位置情報", "カメラ", "マイク"]  # TODO add attentional permissions
            values = [0] * len(keys)  # permission vector
            app_perms_dict = dict(zip(keys, values))  # permisson dict appA → # {A:True,B:False,C:True}
            app_perms = temp.read().split("\n")[2:]
            for app_perm in app_perms:
                if app_perm in app_perms_dict:
                    app_perms_dict[app_perm] = 1  # 権限があれば1，なければ0
        dic[dir] = app_perms_dict
    return dic

def load_pseudo_apps(m):
    # return jpapps,enapps
    corres_dict = make_corres_dict()

    jp_app_files_paths = glob.glob(BASE_DIR + "pseudo_data/*/*/*.tsv")  # 30129

    # ['C:/Users/limin/englishdata,Action,air.com.distractionware.vvvvvvmobile\\air.com.distractionware.vvvvvvmobile.tsv']
    # [いらんやつ,カテゴリ(EN/JP),id,ファイル名]


    jp_apps = list(map(lambda x: x.split("\\"), jp_app_files_paths))
    jp_apps_dict = {}
    for i, jp_app in enumerate(jp_apps):
        cat = corres_dict[jp_app[1]]  # カテゴリ名→カテゴリIDに変換(言語非依存)
        jp_apps_dict[jp_app[2]] = [jp_app_files_paths[i], cat]
    # en_app = ['C:/Users/limin/japanesedata', 'アクション', 'air.com.distractionware.vvvvvvmobile', 'air.com.distractionware.vvvvvvmobile.tsv']
    # corres_dict[en_app[1]] → カテゴリ番号
    # en_app[2] → ID
    # apps → [desc,title,[perm1,perm2,perm3],categoly]
    # categoly → 1～61の数字で表現(言語非依存にするため) → 各 -1 する(配列的な)
    # permissions_texts_dir[26:-4] → ID
    # appfile→ id \t タイトル \t ジャンル \t 説明文

    permissions_texts_dirs = glob.glob(PERMISSION_DIR + TEXT)

    # データの読み込み

    for dir in permissions_texts_dirs:
        # 権限の処理

        with open(dir, "r", encoding="utf-8") as temp:
            keys = ["位置情報", "カメラ", "マイク"]  # TODO attentional permissions
            values = [0] * len(keys)  # permission vector
            app_perms_dict = dict(zip(keys, values))  # permisson dict appA → # {A:True,B:False,C:True}
            app_perms = temp.read().split("\n")[2:]
            for app_perm in app_perms:
                if app_perm in app_perms_dict:
                    app_perms_dict[app_perm] = 1  # 権限があれば1，なければ0
        id = dir[26:-4]
        # 日本語の記述の処理
        if id in jp_apps_dict:
            jp_app_path = jp_apps_dict[id][0]
            with open(jp_app_path, "r", encoding="UTF-8") as app_data:
                app = app_data.read()
            # app = ここでOpen
            app = app.split("\t")
            title = app[1]
            original_desc = app[3]
            jp_apps_dict[id].append(title)
            jp_apps_dict[id].append(original_desc)
            jp_apps_dict[id].append(app_perms_dict)
            parsed_desc = parse_jp(original_desc, m)
            jp_apps_dict[id].append(parsed_desc)


    for k, v in list(jp_apps_dict.items()):
        if len(v) < 3:  # jpdescがないアプリを排除
            jp_apps_dict.pop(k)

    # jp_apps_dict[id]→[path,cat,titile,desc,permidic]
    # key → application_name, value → dict{permission:bool}

    return jp_apps_dict

def load_apps(m,use_zipped_data=False):
    #return jpapps,enapps
    if use_zipped_data and os.path.exists(DATA_FILE) :
        return "apps"
    corres_dict = make_corres_dict()

    en_app_files_paths = glob.glob(BASE_DIR + "englishdata/*/*/*.tsv")
    jp_app_files_paths = glob.glob(BASE_DIR +"japanesedata/*/*/*.tsv")  # 30129

    en_apps = list(map(lambda x: x.split("\\"), en_app_files_paths))
    # ['C:/Users/limin/englishdata,Action,air.com.distractionware.vvvvvvmobile\\air.com.distractionware.vvvvvvmobile.tsv']
    # [いらんやつ,カテゴリ(EN/JP),id,ファイル名]

    en_apps_dict = {}
    for i,en_app in enumerate(en_apps):
        cat = corres_dict[en_app[1]]
        en_apps_dict[en_app[2]] =[en_app_files_paths[i],cat]

    jp_apps = list(map(lambda x: x.split("\\"), jp_app_files_paths))
    jp_apps_dict = {}
    for i, jp_app in enumerate(jp_apps):
        cat = corres_dict[jp_app [1]] #カテゴリ名→カテゴリIDに変換(言語非依存)
        jp_apps_dict[jp_app [2]] = [jp_app_files_paths[i], cat]
    #en_app = ['C:/Users/limin/japanesedata', 'アクション', 'air.com.distractionware.vvvvvvmobile', 'air.com.distractionware.vvvvvvmobile.tsv']
    #corres_dict[en_app[1]] → カテゴリ番号
    #en_app[2] → ID
    #apps → [desc,title,[perm1,perm2,perm3],categoly]
    #categoly → 1～61の数字で表現(言語非依存にするため) → 各 -1 する(配列的な)
    # permissions_texts_dir[26:-4] → ID
    #appfile→ id \t タイトル \t ジャンル \t 説明文


    permissions_texts_dirs = glob.glob(PERMISSION_DIR + TEXT)

    #データの読み込み

    for dir in permissions_texts_dirs:
        #権限の処理

        with open(dir,"r", encoding="utf-8") as temp :
            keys = ["位置情報", "カメラ", "マイク"]  # TODO attentional permissions
            values = [0] * len(keys) #permission vector
            app_perms_dict = dict(zip(keys, values))  # permisson dict appA → # {A:True,B:False,C:True}
            app_perms = temp.read().split("\n")[2:]
            for app_perm in app_perms :
                if app_perm in app_perms_dict:
                    app_perms_dict[app_perm] = 1 #権限があれば1，なければ0
        id = dir[26:-4]
        #日本語の記述の処理
        if id in jp_apps_dict :
            jp_app_path = jp_apps_dict [id][0]
            with open(jp_app_path,"r",encoding="UTF-8") as app_data:
                app = app_data.read()
            # app = ここでOpen
            app = app.split("\t")
            title = app[1]
            original_desc = app[3]
            jp_apps_dict[id].append(title)
            jp_apps_dict[id].append(original_desc)
            jp_apps_dict[id].append(app_perms_dict)
            parsed_desc = parse_jp(original_desc,m)
            jp_apps_dict[id].append(parsed_desc)
        #英語の記述の処理
        if id in en_apps_dict :
            en_app_path = en_apps_dict[id][0]
            app = ""
            with open(en_app_path,"r",encoding="UTF-8") as app_data:
                app = app_data.read()
            # app = ここでOpen
            app = app.split("\t")
            title = app[1]
            desc = app[3]
            parsed_desc = desc.split(" ")
            en_apps_dict[id].append(title)
            en_apps_dict[id].append(parsed_desc)
            en_apps_dict[id].append(app_perms_dict)

    for k,v in list(jp_apps_dict.items()):
        if len(v) <3: #jpdescがないアプリを排除
            jp_apps_dict.pop(k)
    for k,v in list(en_apps_dict.items()):
        if len(v) <3:#endescがないアプリを排除
            en_apps_dict.pop(k)
# jp_apps_dict[id]→[path,cat,titile,desc,permidic]
#key → application_name, value → dict{permission:bool}

    return jp_apps_dict,en_apps_dict

def extract_verb_phrase(app_desc):
    verb_phrases = []
    return verb_phrases
def make_corres_dict():
    corres_dict = {}
    with open(BASE_DIR + "eandj.txt") as f:
        temps= f.read().split("\n")
        temps = list(map(lambda x: x.split(","), temps))
        for temp in temps[:-1]: #なんかラストが駄目っぽいので-1
            print(temp)
            corres_dict[temp[1]] = int(temp[0]) -1 # -1 → 配列の関係
            corres_dict[temp[2]] = int (temp[0]) -1
    return corres_dict


# 権限と単語に関する相互情報量を求めたい M(word,p) = d(word,p) / (d(word) * d(p))
# doc = {categoly : c1,permssion:[p1,p2],description:[w1,w2,w3]}
# 日本語データを使ったときとそうでないときの差を見てみる
# targetP => 権限を指定()
def calc_mi(docs, targetP, K=50):
    N11 = defaultdict(lambda: 0)  # N11[word] -> wordを含むPを含む文書数
    N10 = defaultdict(lambda: 0)  # N10[word] -> wordを含むPを含まない文書数
    N01 = defaultdict(lambda: 0)  # N01[word] -> wordを含まないPを含む文書数
    N00 = defaultdict(lambda: 0)  # N00[word] -> wordを含まないPを含まない文書数
    Np = 0
    Nn = 0
    V = set()
    for doc in docs.values():
        description = doc[WAKATI]
        if doc[PERM][targetP] == 1:
            for word in description:
                Np += 1
                N11[word] += 1
                V.add(word)
        else:
            for word in description:
                Nn += 1
                N10[word] += 1
                V.add(word)
    for word in V:
        N01[word] = Np - N11[word]
        N00[word] = Nn - N10[word]

    N = Np + Nn

    # 各単語の相互情報量を計算
    MI = []
    for word in V:
        n11, n10, n01, n00 = N11[word], N10[word], N01[word], N00[word]
        # いずれかの出現頻度が0.0となる単語はlog2(0)となってしまうのでスコア0とする
        if n11 == 0.0 or n10 == 0.0 or n01 == 0.0 or n00 == 0.0:
            MI.append((0.0, word))
            continue
        # 相互情報量の定義の各項を計算
        temp1 = n11 / N * math.log((N * n11) / ((n10 + n11) * (n01 + n11)), 2)
        temp2 = n01 / N * math.log((N * n01) / ((n00 + n01) * (n01 + n11)), 2)
        temp3 = n10 / N * math.log((N * n10) / ((n10 + n11) * (n00 + n10)), 2)
        temp4 = n00 / N * math.log((N * n00) / ((n00 + n01) * (n00 + n10)), 2)
        score = temp1 + temp2 + temp3 + temp4
        MI.append((score, word))

    # 相互情報量の降順にソートして上位k個を返す
    MI.sort(reverse=True)
    return MI[:K]


def load_answer(use_zipped_data=False):
    ID = 0
    ANSWER_LOC = 2
    ANSWER_MIKE = 3
    ANSWER_CAMERA = 4

    # dic[id] = 正解の記述
    # (id).txt → 正解
    dic = {}
    dir = BASE_DIR + ANSWER_DIR
    answer_paths = glob.glob(dir + "/*.txt")
    for path in answer_paths:
        with open(path,"r") as temp:
            answers = temp.read().split("\n")
            dic[answers[0]] = [answers[ANSWER_LOC],answers[ANSWER_MIKE],answers[ANSWER_CAMERA]]
    return dic

def make_word_to_norm(w2vmodel):
    word_to_norm_dict = {}
    w_vector = w2vmodel.wv
    voc =  list(w_vector.vocab)
    for word in voc:
        word_to_norm_dict[word] = np.linalg.norm(w_vector[word])
    return word_to_norm_dict
# def load_pseudo_apps(en_apps,use_zipped_data=False):
#     pseudo_jp_apps = []
#     translator = Translator()
#     for en_app in en_apps.values():
#         try:
#             desc = " ".join(en_app[DESC])
#             pseudo_jp_app = copy.copy(en_app)
#             pseudo_jp_app[DESC] = translator.translate(desc, dest='ja').text
#             pseudo_jp_apps.append(pseudo_jp_app)
#
#         # phase0: カテゴリ間の類似度,tf-idf,idf,tfなどの計算
#         except:
#             traceback.print_exc()
#     return pseudo_jp_apps
def parse_jp(text,t=None):
    日本語に対してNoneを返す正規表現 = re.compile('[ぁ-んァ-ン一-龥ー。、．，· :]+')
    HINSI = 3
    HINSIDETAIL = 1
    GENKEI = 2
    if t == None:
        t  = MeCab.Tagger("-Ochasen")
    parsed = t.parse(text)
    parsed = parsed.split("\n")
    text_parsed = []
    parsed = parsed[:-2]
    for word in parsed:
        splited = word.split("\t")
        hinsi_splited = splited[3].split("-")
        hinsi_main =  hinsi_splited[0]
        genkei_word = splited[GENKEI]
        if (hinsi_main == "名詞" or hinsi_main == "動詞") and (日本語に対してNoneを返す正規表現.match(genkei_word) ):
            text_parsed.append(genkei_word)
    return text_parsed

def calc_cos_sim_using_normdic(w1,w2,v1,v2,normdic,dot_dic_w2v,wtoind):
     return calc_cos_sim(v1,v2,n1=normdic[w1],n2=normdic[w2],dot = dot_dic_w2v[wtoind[w1]][wtoind[w2]])

def make_word_dot_dict(words,w2vmodel):
    d = {}
    wv = w2vmodel.wv
    for w1 in words:
        dd = {}
        for w2 in words:
            dd[w2] = np.dot(wv[w1],wv[w2])
        d[w1] = dd
    return d #d[w1][w2]で内積を得る


def calc_cos_sim(v1, v2,n1 = None,n2 = None,dot = None):
    if n1 == None:
        n1 = np.linalg.norm(v1)
    if n2 == None:
        n2 = np.linalg.norm(v2)
    if dot ==None:
        dot = np.dot(v1, v2)
    result = dot / (n1* n2)
    if np.isnan(result):
        result = 0
    return result

def calc_ndl(docs):
    dls = list(map(lambda x: len(x), docs))
    ave = sum(dls)/len(dls)
    ndls = list(map(lambda x:x/ave,dls))
    return ndls

def make_tfidf(tf,idf):
    tf_idf  = collections.defaultdict(lambda :collections.defaultdict(lambda :0))
    for word,v in idf.items(): #idf[word] → wordのidf
        for index,j in enumerate(tf): #tf[w] → index番目の文章のwのtf
            tf_idf[index][word] = v * j[word] # index 番目の文章のwordのtf-idf

    return tf_idf # key→(doc_index,word)

def make_idf(docs): #docs→分かち書き済みのdoc["りんご","食べる"]s
    dic = defaultdict(lambda :0)
    for doc in docs:
        temp = list(set(doc))
        for word in temp:
            dic[word] += 1
    size = len(docs)
    idf = {}
    for k,v in dic.items():
        idf[k] = math.log2(size/(dic[k]+1))
    return idf

def calc_tf(doc): # doc → [わかち済み ]
    tf_dic = defaultdict(lambda :0)
    for word in doc:
        tf_dic[word]+=1
    l = len(doc)
    for k,v in tf_dic.items():
        tf_dic[k]= v/l
    return tf_dic # tf[word] → ある文のwordのtf値


def make_tf_idf_ndl(docs):
    #i→単語 j→文章のindex
    #ndl → dl/average(dls)
    tf = defaultdict(lambda: defaultdict(lambda: 0))
    idf =  defaultdict(lambda: defaultdict(lambda: 0))
    dl = []
    idf_result = {}
    N =len(docs)
    for index,doc in enumerate(docs):
        dl.append(len(doc))
        for word in doc:
            tf[index][word] += 1 #i番目の文章に含まれる単語
            idf[word][index] = 1 #あるwordをkeyとした辞書のTrueの数

    for key,i in dict(idf).items():
        idf_result[key]  = (sum(i.values()))
        #idf_result[word] → 出てきた文章すうが返る

    for index,doc_tf in enumerate(tf):
        for word in doc_tf:
            doc_tf[word]=tf[index][word]/dl[index]
    LogN = math.log2(N)
    for word,v in idf_result:
        idf_result[word] = LogN - math.log2(v)
    avedl = sum(dl)/N
    ndl = list(map(lambda x:x/avedl,dl))
    return tf,idf_result,ndl
    #tf[index][word]
    #idf[word]
    #[[cat1 tf-idf_vec],[cat2 tf-idf_vec],[cat3 tf-idf_vec]]
#docs → [[[w1,w2],[w3,w4],[w5,w6]] [[w1,w4],[w5,w6],[w7,w2]]]
#[[[カテゴリ1の文1],[カテゴリ1の文2],[カテゴリ1の文3# ]],
# [[カテゴリ2の文1],[カテゴリ2の文2],[カテゴリ2の文3]]]

def original_text_to_parsed(text,fil = None): # fil→フィルター 品詞とか単語帳とか言語でフィルタリング
    m = MeCab.Tagger("-Ochasen")
    parsed_text = list(map(lambda x: x.split("\t"), m.parse(text).split("\n")[:-2]))
    if fil == None:
        parsed_text = list(map(lambda x:x[2],parsed_text)) #原形を返す
        return parsed_text
    parsed_text_filtered = fil(parsed_text)

    return parsed_text_filtered



def my_filter (parsed_and_postagged,nokosu_hinsi = []):

    """
文の中でいらないものをフィルターする

        Parameters
        ----------
     parsed_and_postagged : List<List>
        分かち書き済みの文: [[わたし,ワタシ,わたし,名詞-代名詞-一般,"",""],[わたし,ワタシ,わたし,名詞-代名詞-一般,"",""],[わたし,ワタシ,わたし,名詞-代名詞-一般,"",""]]

     nokosu_hinsi: List<String>
        残す品詞のリスト:["動詞","名詞"]


        Returns
        -------
       flitered_text :  List<String>
        フィルタリング条件 e.g 単語長さ,品詞,単語長さ
        でフィルター済みの文,原形のみを詰め込んだ:

        """
    nokosu_hinsi_dict ={x : True for x in nokosu_hinsi }
    #parsed_and_postagged =
    result = []

    for word in parsed_and_postagged:
        if nokosu_hinsi_dict[word[POSTAG]]:
            result.append(word[GENKEI])

    return result


#format
# ファイル名:id.txt
#中身:
"""
id
権限1の答え
権限2の答え
権限3の答え
"""

def load_answers():
    paths = glob.glob(ANSWER_DIR + "/*")
    print(paths)
    di = {}
    for path in paths:
        with open(path,"r") as f:
            temp = f.read().split("\n")

            print(path)
            di[temp[0]] = [temp[1],temp[2],temp[3]]
    return di

def merge_2_cat(c1,c2):
    newc = defaultdict(lambda :{})
    for k1,v1 in c1.items():
        for k2,v2 in v1.items():
            newc[k1][k2] = (c1[k1][k2] + c2[k1][k2])/2
    return newc

def load_dot_dic(): #w1とw2でその内積を返す辞書

    dot_dic =   np.load("npsave2.npy")
    with open("vtoind.pickle","rb") as f:
        vtoind = pickle.load(f)
    with open("allvocab.pickle", "rb") as f:
        all_v = pickle.load(f)
    dic = {}
    all_v = list(all_v)
    for v1 in all_v[:-1]:
        dic2 = {}
        for v2 in all_v[:-1]:
            i1 = vtoind[v1]
            i2 = vtoind[v2]
            dic2[v2] = dot_dic[i1][i2]
        dic[v1] = dic2
    return dic

def make_dot_dic(): #ind(w)２つでそのdotを返す配列を作る巻数

    with open("allvocab.pickle", "rb") as f:
        all_v = pickle.load(f)
    # with open("allvocab.pickle", "wb") as f:
    #     pickle.dump(all_v,f)

    w2_model = gensim.models.Word2Vec.load('ja/ja.bin')
    wv = w2_model.wv
    l = len(all_v)
    dot_dic = [-1] *l
    vtoind = dict(zip(all_v, list(range(l)))) #単語から数値に直す辞書 これによりピックルできるようにする
    for i, v1 in enumerate(all_v):
        tempdic = [-1] * (l)
        for j, v2 in enumerate(all_v[i:]):
            id2 = vtoind[v2]
            tempdic[id2] = np.dot(wv[v1], wv[v2])
        id1 = vtoind[v1]
        dot_dic[id1] = tempdic
        if i % 10 == 0:
            print(str(i) + "/" + str(l))
    # with open("dot_dic.pickle", "wb") as f:
    #     pickle.dump(dot_dic, f)
    return dot_dic


def random_pick_apps(N=30):
    with open("jp_apps_dict.pickle", "rb") as f:
        jp_apps = pickle.load(f)
    apps_per_perms = []
    for i in range(3):
        apps_per_perm = []
        for app in jp_apps.values():
            if app[PERM][i] == 1:
                apps_per_perm.append(app) #TODO ここIDで

        l = len(apps_per_perm)
        temp = random.sample(apps_per_perm,N)
        apps_per_perms.append(temp)
    return apps_per_perms
    #これで30*3の90アプリが選出される これを対象として評価

def get_cos_sim_between_2words():
    w2_model = gensim.models.Word2Vec.load('ja/ja.bin')
    dd = np.load("words2dot.npy")
    normdic_w2v = make_word_to_norm(w2_model)
    w2ind = np.load("vtoind.pickle")
    vocab = list(w2ind.keys())
    l = len(vocab)
    cos_sim = []
    for i,v1 in enumerate(vocab):
        print(str(i)+"/"+str(l))
        temp1 =[]
        for v2 in vocab:
            id1 = w2ind[v1]
            id2 = w2ind[v2]
            dot = dd[id1][id2]
            if dot == -1:
                dot = dd[id2][id1]
            n1 = normdic_w2v[v1]
            n2 = normdic_w2v[v2]
            result = dot / (n1 * n2)
            temp1.append(result)
        cos_sim.append(temp1)
    return cos_sim #cos_sim[wtoind[w1]][wtoind[w2]] → cos sim between two words
