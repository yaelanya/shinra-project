import numpy as np
import pandas as pd
import re
import json
import MeCab
import difflib
import wikitextparser as wtp


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

def _sub(text, s):
    for i in s:
        if isinstance(i, str):
            text = text.replace(i, '')
        else:
            text = text.replace(i.string, '')
    
    return text

def _clean_source_text(parsed_source_text):
    clean_text = _sub(parsed_source_text.contents, parsed_source_text.templates)
    clean_text = _sub(clean_text, parsed_source_text.tags())
    clean_text = _sub(clean_text, parsed_source_text.external_links)
    clean_text = re.sub(r'\n|\t|\r', ' ', clean_text)
    clean_text = re.sub(r'={2,}.*?={2,}', '', clean_text)
    clean_text = re.sub(r'\[\[[^\]]+:.+?\]\]', '', clean_text)
    clean_text = re.sub(r'\[\[[^\]]+?\||\]\]|\[\[', '', clean_text)
    clean_text = re.sub(r'\'{2,}|\*+|#+', '', clean_text)
    clean_text = re.sub(r'<[^>]*?>.*?<\/[^>]*?>', '', clean_text)
    clean_text = re.sub(r'\{\{.*?\}\}|\{.*?\}', '', clean_text)
    
    return clean_text

def _complement_subtitle(article_df: pd.DataFrame):
    # サブカテゴリ名の無い部分のうち，先頭部分は　NO_SUBTITLE で埋める
        if len(article_df.loc[article_df.heading != '']) is 0:
            article_df.loc[article_df.heading == '', ['heading']] = 'NO_SUBTITLE'
        else :
            article_df.loc[0:article_df[article_df.heading != ''].index[0], ['heading']] = 'NO_SUBTITLE'
        
        # サブカテゴリ名が無い場合は，1つ前のサブカテゴリ名で補完する
        while len(article_df.loc[article_df.heading == '']) > 0: 
            article_df.loc[article_df.heading == '', ['heading']] = \
                article_df.loc[article_df.loc[article_df.heading == '', ['heading']].index - 1, 'heading'].values[0]

        return article_df

def _search_subtitle(source_text: str):
    m = re.search(r'==+\s*([^=]+)\s*==+', source_text)
    if m:
        heading = m.group(1)
    else:
        heading = ''

    return heading

def _get_subtitle_of_sentence(article_df: pd.DataFrame, source_text: str, heading: str):
    df = article_df.copy()
    for s in re.findall(r'.*?。', source_text):
        m_sentence = difflib.get_close_matches(s.strip(), df.sentence.values, n=1)
        if len(m_sentence) > 0 and len(heading) > 0:
            # heading にサブタイトル名を追加
            df.loc[df.sentence == m_sentence[0], ['heading']] += heading + ','
    
    return df

def get_subtitle(sentence_df: pd.DataFrame, wiki_dump_data: list):
    df = sentence_df.assign(heading = '')
    new_train_df = pd.DataFrame()
    for _id in df._id.unique():
        article_df = df.loc[df._id == _id]
        
        row_article = [entry for entry in wiki_dump_data if entry['index']['_id'] == _id][0]
        parsed = wtp.parse(row_article['source_text'])
        for source in parsed.sections[1:]:
            heading = _search_subtitle(source.string)
            section_text = _clean_source_text(source)
            article_df = _get_subtitle_of_sentence(article_df, section_text, heading)
        
        article_df = _complement_subtitle(article_df)
        new_train_df = new_train_df.append(article_df)

    return new_train_df

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

def is_oxidation_state_parts(surface):
    return re.match(r'^\($|^[IV]{1,4}$|^\)$', surface)
    
def is_oxidation_state(word):
    return re.match(r'^\([IV]{1,4}\)$', "".join(word))

def get_compound_list(text: str):
    mecab_param = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
    mecab_param.parse("")
    node = mecab_param.parseToNode(text)
    
    compound_list = []
    compound = []
    oxidation_state = []
    while node:
        if len(node.surface) == 0:
            node = node.next
            continue
        
        hinshi = node.feature.split(',')    
        if is_oxidation_state_parts(node.surface):
            oxidation_state.append(node.surface)
        elif (len(oxidation_state) > 0) and is_oxidation_state(oxidation_state):            
            compound.append(''.join(oxidation_state))
            oxidation_state = []
            
        if is_noun2(hinshi):
            compound.append(node.surface)
        elif len(compound) > 0 and not is_oxidation_state_parts(node.surface):
            compound_list.append(''.join(compound))
            compound = []
        
        node = node.next
    
    if len(compound) > 0:
        if is_oxidation_state(oxidation_state): compound += oxidation_state
        compound_list.append(''.join(compound))

    return compound_list

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
