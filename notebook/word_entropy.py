import numpy as np
import pandas as pd
from collections import Counter

import shinra_util as util

def word_proba(docs: [pd.Series, list], word: str):
    word_count = np.array([Counter(doc)[word] for doc in docs])
    return pd.Series(word_count / np.sum(word_count))

def entropy(word_proba: pd.Series):
    return np.sum(np.nan_to_num(-word_proba * np.log2(word_proba)))

def word_entropy(clue_word_df: pd.DataFrame, normalize=True):
    word_set = list(set(util.flatten(clue_word_df['clue_word'].values)))
    Hp = np.array([clue_word_df.loc[clue_word_df.label == True, 'clue_word'].pipe(word_proba, word).pipe(entropy) for word in word_set])
    Hn = np.array([clue_word_df.loc[clue_word_df.label == False, 'clue_word'].pipe(word_proba, word).pipe(entropy) for word in word_set])
    if normalize: 
        Hp /= np.log2(len(clue_word_df.loc[clue_word_df.label == True]))
        Hn /= np.log2(len(clue_word_df.loc[clue_word_df.label == False]))

    return pd.DataFrame({"clue_word": word_set, "entropy_positive": Hp, "entropy_negative": Hn})