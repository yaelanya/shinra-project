'''
reference : https://blog.spot-corp.com/other/2016/07/19/cabocha_nlp.html
'''

import pandas as pd
import MeCab
import CaboCha

class DependencyParsing(object):
    def __init__(self, args=None):
        if args is None:
            args = '-f1 -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd'
        
        self.cp = CaboCha.Parser(args)
        
    def parsing(self, sentence):
        tree = self.cp.parse(sentence)
        tokens = self._to_tokens(tree)

        head_tokens = list(filter(self._has_chunk, tokens))
        words = map(lambda x: x.surface, head_tokens)

        lasts = self._chunk_by(self._has_chunk, tokens)

        links = map(lambda x: x.chunk.link, head_tokens)
        link_words = map(lambda x: self._concat_tokens(x, head_tokens, lasts), links)

        to_words = []
        from_words = []
        for (i, to_word) in enumerate(link_words):
            to_words.append(to_word)
            from_words.append(self._concat_tokens(i, head_tokens, lasts))
            
        return pd.DataFrame({"from_word": from_words, "to_word": to_words})
        
    def _chunk_by(self, func, col):
        '''
        `func`の要素が正のアイテムで区切る
        '''
        result = []
        for item in col:
            if func(item):
                result.append([])
            else:
                result[len(result) - 1].append(item)
        return result

    def _has_chunk(self, token):
        '''
        チャンクがあるかどうか
        チャンクがある場合、その単語が先頭になる
        '''
        return token.chunk is not None

    def _to_tokens(self, tree):
        '''
        解析済みの木からトークンを取得する
        '''
        return [tree.token(i) for i in range(0, tree.size())]

    def _concat_tokens(self, i, tokens, lasts):
        '''
        単語を意味のある単位にまとめる
        '''
        if i == -1:
            return None
        word = tokens[i].surface
        last_words = map(lambda x: x.surface, lasts[i])
        
        return word + ''.join(last_words)