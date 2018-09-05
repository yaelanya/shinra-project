import MeCab

m = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

'''
clue_words = []
for i, v in train.iterrows():
    node = m.parseToNode(v['sentence'])
    clue_word = []
    while node:
        if len(node.surface) is 0:
            node = node.next
            continue
        
        hinshi = node.feature.split(',')
        if hinshi[0] == '名詞' and hinshi[1] == 'サ変接続':
            clue_word.append(node.surface)
        
        node = node.next
        
    clue_words.append(clue_word)
'''