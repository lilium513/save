# Howto? → clacジャカード係数
# JD → AnB / AuB
from tools import calc_jaccard_distance


def calc_perm_sim(p1, p2): #TODO 対象の権限を増やす

    if len(p1) !=  len(p2):
        return 0

    else:
#        p1 [0,1,0] or p1 ["カメラ","マイク"]
        return calc_jaccard_distance(p1,p2)