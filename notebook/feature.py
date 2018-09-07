import numpy as np
import pandas as pd
import re

import shinra_util as util

def contains_clue_word(train_df: pd.DataFrame, clue_word: list):
    feature_word_df = pd.DataFrame()

    for word in clue_word:
        feature_word_df[word] = train_df.sentence.str.contains(f"{word}").tolist()

    return feature_word_df

def subtitle_cat(train_df: pd.DataFrame, clue_word: list):
    # サブタイトル名をもとにカテゴリ変数を作成する
    df = train_df.assign(heading_cat = np.nan)
    df.loc[df.heading.str.contains(r'NO_SUBTITLE'), 'heading_cat'] = 0
    df.loc[df.heading.str.contains(util.contains_patt(clue_word)), 'heading_cat'] = 1
    df.loc[df.heading_cat.isna(), 'heading_cat'] = 2

    return df.heading_cat.astype('category')
