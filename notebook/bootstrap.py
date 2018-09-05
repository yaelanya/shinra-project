import numpy as np
import pandas as pd
import MeCab
from collections import Counter

import shinra_util as util

class Bootstrap(object):
    def __init__(self, parsed_df: pd.DataFrame):
        self.m = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
        self.noun_list = []
        self.clue_words = []
        self.parsed_df = parsed_df

    def run(self, init_clue_words: list, top_n=50):
        loop_count = 0
        n_old_clue_words = 0
        self.clue_words = init_clue_words
        while True:
            new_noun_list = \
            util.flatten(
                self.parsed_df.loc[self.parsed_df.to_word.isin(self.clue_words)].apply(
                    lambda x: util.get_noun_list(x.from_word)
                    , axis=1
                ).tolist()
            )
            self.noun_list = list(set(self.noun_list + new_noun_list))

            patt = util.contains_patt(self.noun_list)
            clue_words_count = Counter(self.parsed_df.loc[self.parsed_df.from_word.str.contains(patt)].to_word.tolist()).most_common()
            new_clue_words = np.array([np.array([key, value]) for key, value in clue_words_count])[:top_n, 0]
            self.clue_words = list(set(self.clue_words + list(new_clue_words)))

            loop_count += 1
            print("loop: ", loop_count)
            print("clue words count: ", len(self.clue_words))

            if n_old_clue_words == len(self.clue_words):
                break
            else:
                n_old_clue_words = len(self.clue_words)