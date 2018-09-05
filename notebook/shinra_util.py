import numpy as np
import pandas as pd
import re
import json
import MeCab


def read_jasonl(filename):
    with open(filename) as f:
        return [json.loads(line.rstrip('\r\n')) for line in f.readlines()]

def flatten(multi_list: list):
    return [item for sublist in multi_list for item in sublist if (not isinstance(item, str)) or (len(item) is not 0)]

def clean_text(text: str):
    cleaned = re.sub(r'\\[a-zA-Z0-9]+', '', text)
    cleaned = re.sub(r'{.+}', '', cleaned)
    cleaned = re.sub(r'\"', '', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    
    return cleaned

def text2sentence(text: str):
    if re.search(r'。', text):
        return list(map(lambda s: s.strip(), re.findall(".*?。", clean_text(text))))
    else:
        return [clean_text(text)]

def contains_patt(match_text: [str, list]):
    if isinstance(match_text, str):
        return f"{match_text}".replace(r'(', r'\(').replace(r')', r'\)')
    elif isinstance(match_text, list):
        return "|".join(match_text).replace(r'(', r'\(').replace(r')', r'\)')
    else:
        print("Unexpected type.")
        return ""

def get_noun_list(text: str, join=True):
    mecab_param = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
    mecab_param.parse("")
    node = mecab_param.parseToNode(text)
    
    noun_list = []
    noun = []
    while node:
        if len(node.surface) == 0:
            node = node.next
            continue

        hinshi = node.feature.split(',')
        if hinshi[0] in ['名詞', '接頭詞']:
            if join:
                noun.append(node.surface)
            else:
                noun_list.append(node.surface)
        elif (len(noun) > 0) and join:            
            noun_list.append(''.join(noun))
            noun = []
        
        node = node.next
    
    if (len(noun) > 0) and join:
        noun_list.append(''.join(noun))

    return noun_list
