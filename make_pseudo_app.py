#1. 英語データの読み込み
#2. 翻訳を例外が出るまで続ける
import time

from googletrans import Translator

import glob
import tools
import os
import traceback
import sys
import re
def make_pseudo():
    translator = Translator()

    # with open("english_apps_paths_for_pseudo.txt","r") as f:
    #     en_app_files_paths = f.read().split("\n")
    psudo_path = tools.BASE_DIR + "pseudo_data"
    en_app_files_paths = make_pseudo_paths()
    pseudo = []
    pattern = re.compile(r'[^a-zA-Z0-9 .,!?]')
    for num,i in enumerate(en_app_files_paths):
        loop_flag = True
        if num % 10 == 0:
            print(num)
        while loop_flag:
            try:
                with open(i,"r",encoding="UTF-8") as f:
                    app = f.read()
                    app = app.split("\t")
                    text = pattern.sub(" ",app [3])
                    app [3] = translator.translate(text, dest='ja').text
                    filename = "\\" + app[0] + ".tsv"
                    dirname = psudo_path +"\\" +app[2] +"\\"+app[0]
                    os.makedirs(dirname, exist_ok=True)
                    pseudo.append(app)
                with open( dirname+ filename, "w",encoding="UTF-8") as ff:
                    ff.write("\t".join(app))
                loop_flag = False
                time.sleep(1)
            except KeyboardInterrupt:
                traceback.print_exc()
                sys.exit(1)

            except:
                traceback.print_exc()
                print(app [3])
                time.sleep(5)
                loop_flag = False

def make_pseudo_paths():
    already_get = glob.glob(tools.BASE_DIR + "pseudo_data/*/*/*")
    pseudo_already_get_id = list(map(lambda x:x.split("\\")[-1][:-4],already_get))
    with open("english_apps_paths_for_pseudo.txt", "r") as f:
        en_app_files_paths = f.read().split("\n")
    en_app_files_ids = list(map(lambda x:x.split("\\")[-1][:-4],en_app_files_paths))
    en_app_files_dict = dict(zip(en_app_files_ids,en_app_files_paths))
    for id in pseudo_already_get_id:
        if id in en_app_files_dict:
            en_app_files_dict.pop(id)
    return list(en_app_files_dict.values()) #return:取得したいen_appのpathのList


def init_eng():
    english_paths = glob.glob(tools.BASE_DIR + "englishdata/*/*/*")
    with open("english_apps_paths_for_pseudo.txt","w") as f:
        f.write("\n".join(english_paths))
    # english_id = list(map(lambda x: x.split("\\")[-1][:-4], english_paths))
    # eng_dic = zip(english_id,english_paths)
    # return eng_dic