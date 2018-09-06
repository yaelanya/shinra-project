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

def train2dict(train_data: list, attribute: str):
    train_dict = {}
    for entry in train_data:
        if len(entry['Attributes'][attribute]) is 0: continue
        train_dict[str(entry['WikipediaID'])] = flatten([text2sentence(item) for item in entry['Attributes'][attribute]])

    return train_dict

def labeling(sentence_df: pd.DataFrame, train_dict: dict):
    _sentence_df = sentence_df.assign(label = False)
    for _id, train_str in train_dict.items():
        _sentence_df.loc[_sentence_df._id == str(_id), 'label'] = \
            _sentence_df.loc[_sentence_df._id == str(_id)].sentence.str.contains(contains_patt(train_str))

    return _sentence_df

def is_noun1(hinshi: list):
    if hinshi[0] in ['名詞', '接頭詞']:
        return True
    else:
        return False

def is_noun2(hinshi: list):
    if (hinshi[0] == '名詞') and (hinshi[1] == '固有名詞') and (hinshi[2] != '一般'):
        return False
    elif (hinshi[0] == '名詞') and (hinshi[1] in ['代名詞', '非自立', '特殊']):
        return False
    elif hinshi[0] == '名詞' or (hinshi[0] == '接頭詞' and hinshi[1] == '名詞接続'):
        return True
    else:
        return False

def is_noun3(hinshi, noun):
    if not (hinshi[0] in ['名詞', '接頭詞']) and (len(noun) == 0):
        return False
    elif (hinshi[0] == '名詞') and (hinshi[1] == '固有名詞') and (hinshi[2] != '一般'):
        return False
    elif (hinshi[0] == '名詞') and (hinshi[1] in ['代名詞', '非自立', '特殊']):
        return False
    elif (hinshi[0] in ['名詞', '接頭詞']) or ((hinshi[0] == '助詞') and (hinshi[1] in ['連体化', '並立助詞', '副助詞'])):
        return True
    else:
        return False

def get_noun_list(text: str, join=True, condition=2):
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

        if condition is 1: is_noun = is_noun1(hinshi)
        elif condition is 2: is_noun = is_noun2(hinshi)
        elif condition is 3: is_noun = is_noun3(hinshi, noun)
        else: is_noun = False

        if is_noun:
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

def get_word_list(text: str, condition_func=None):
    mecab_param = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
    mecab_param.parse("")
    node = mecab_param.parseToNode(text)
    
    words = []
    while node:
        if len(node.surface) == 0:
            node = node.next
            continue

        hinshi = node.feature.split(',')
        if condition_func(hinshi):
            words.append(node.surface)
                   
        node = node.next
    
    return words
