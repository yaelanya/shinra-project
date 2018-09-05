import pandas as pd
import CaboCha
import itertools


'''
reference: https://blog.spot-corp.com/other/2016/07/19/cabocha_nlp.html
'''
class DependencyParsing(object):
    def __init__(self):
        self.cp = CaboCha.Parser('-f1 -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

    def chunk_by(self, func, col):
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

    def has_chunk(self, token):
        '''
        チャンクがあるかどうか
        チャンクがある場合、その単語が先頭になる
        '''
        return token.chunk is not None

    def to_tokens(self, tree):
        '''
        解析済みの木からトークンを取得する
        '''
        return [tree.token(i) for i in range(0, tree.size())]

    def concat_tokens(self, i, tokens, lasts):
        '''
        単語を意味のある単位にまとめる
        '''
        if i == -1:
            return None
        word = tokens[i].surface
        last_words = map(lambda x: x.surface, lasts[i])
        return word + ''.join(last_words)

    def parsing(self, sentence):
        tree = self.cp.parse(sentence)
        tokens = self.to_tokens(tree)
        head_tokens = list(filter(self.has_chunk, tokens))
        lasts = self.chunk_by(self.has_chunk, tokens)
        links = map(lambda x: x.chunk.link, head_tokens)

        link_words = list(map(lambda x: self.concat_tokens(x, head_tokens, lasts), links))
        from_words = [self.concat_tokens(i, head_tokens, lasts) for i, to_word in enumerate(link_words)]
        
        parsed_df = pd.DataFrame({"from_word": from_words, "to_word": link_words})
        parsed_df = parsed_df.loc[parsed_df.to_word.notna()]
        parsed_df.to_word = parsed_df.to_word.str.replace(r'、|。', '')
        parsed_df.from_word = parsed_df.from_word.str.replace(r'、|。', '')

        return parsed_df