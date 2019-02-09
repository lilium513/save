import tools
import pytextrank
import sys
import gensim.summarization.keywords
from tools import original_text_to_parsed
def vote_to_featuring_word(texts, idf, featuring_words, m): #単語に投票フェイズ 権限が関係あるやつ
    votes = {}
    K = 0
    for word in featuring_words :
        votes[word] = 0
    for text in texts:
        text = tools.parse_jp(text,t=m)
        text_with_space =  " ".join(text)
        temp = gensim.summarization.keywords(text_with_space, m,ratio = 1, deacc=False, split=True,scores = True, pos_filter=('NN', 'JJ','VB'))
        text_rank_dic = temp
        if len(text_rank_dic) == 0:
            continue
        max_text_rank = max(text_rank_dic.values())
        for word in text:
            if word in votes and word in text_rank_dic:
                votes[word]+= text_rank_dic[word]/max_text_rank
                K+=1

    for word in featuring_words :
        if word in idf:
            votes[word] *= (idf[word]/K)
    return votes

def vote_to_word(texts, idf, m): #単語に投票フェイズ 権限が関係ないやつ
    votes = {}
    K = 0
    for text in texts:
        text = tools.parse_jp(text,t=m)
        text_with_space =  " ".join(text)
        temp = gensim.summarization.keywords(text_with_space, m,ratio = 1, deacc=False, split=True,scores = True, pos_filter=('NN', 'JJ','VB'))
        text_rank_dic = temp
        if len(text_rank_dic) == 0:
            continue
        max_text_rank = max(text_rank_dic.values())
        for word in text:
            if word in votes and word in text_rank_dic:
                votes[word]+= text_rank_dic[word]/max_text_rank
                K+=1

    for word in votes:
        if word in idf:
            votes[word] *= (idf[word]/K)
    return votes

def vote_to_sentence(sentence,votes_to_word_dict):
    vote = 0
    length_s = len(sentence)
    for word in sentence:
        if word in votes_to_word_dict:
            vote += votes_to_word_dict[word]
    if length_s == 0:
        return 0
    return vote/length_s