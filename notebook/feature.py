import numpy as np
import pandas as pd
import re

import shinra_util as util

def contains_clue_word(train_df: pd.DataFrame, clue_word: list):
    feature_word_df = pd.DataFrame()

    for word in clue_word:
        feature_word_df[word] = train_df.sentence.str.contains(f"{word}").tolist()

    return feature_word_df

def subtitle_cat(train_df: pd.DataFrame):
    # サブタイトル名をもとにカテゴリ変数を作成する
    train = train.assign(heading_cat = np.nan)

    cat1 = r'NO_SUBTITLE'
    train.loc[train.heading.str.contains(cat1), 'heading_cat'] = 0

    cat2 = r'|'.join(np.append(clue_word_entropy, ['用途', '効果', '目的']))
    train.loc[train.heading.str.contains(cat2), 'heading_cat'] = 1

    train.loc[train.heading_cat.isna(), 'heading_cat'] = 2

    train.heading_cat = train.heading_cat.astype('category') 
