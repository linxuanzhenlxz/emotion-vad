import os, re, csv, json, sys, string
import numpy as np
import pandas as pd


from collections import defaultdict, Counter

import gzip

from tqdm import tqdm

import pickle as pkl
from argparse import ArgumentParser
import logging

tqdm.pandas()

parser = ArgumentParser()
parser.add_argument('--dataPath', help='path to CSV data file with texts')
parser.add_argument('--lexPath', help='path to lexicon. CSV with columns "word" plus emotion columns')
parser.add_argument('--lexNames', nargs="*", type=str, help='Names of the lexicons/column names in the lexicon CSV')
parser.add_argument('--savePath', help='path to save folder')

def read_lexicon(path, LEXNAMES):
    df = pd.read_csv(path)
    df = df[~df['word'].isna()]
    df = df[['word']+LEXNAMES]
    df['word'] = [x.lower() for x in df['word']]
    return df
        # df = df[~df['val'].isna()]

def prep_dim_lexicon(df, dim):
        ldf = df[['word']+[dim]]
        ldf = ldf[~ldf[dim].isna()]
        ldf.drop_duplicates(subset=['word'], keep='first', inplace=True)
        ldf[dim] = [float(x) for x in ldf[dim]]
        ldf.rename({dim: 'val'}, axis='columns', inplace=True)
        ldf.set_index('word', inplace=True)
        return ldf

def get_alpha(token):
        return token.isalpha()



def get_vals(twt, lexdf_valence, lexdf_arousal, lexdf_dominance):
        tt = twt.lower().split(" ")
        at = [w for w in tt if w.isalpha()]

        # 找到匹配的词
        pw_valence = [x for x in tt if x in lexdf_valence.index]
        pw_arousal = [x for x in tt if x in lexdf_arousal.index]
        pw_dominance = [x for x in tt if x in lexdf_dominance.index]

        print(f"Text: {twt}, Valence Matches: {pw_valence}, Arousal Matches: {pw_arousal}, Dominance Matches: {pw_dominance}")

        # 提取对应的情感分值
        pv_valence = [lexdf_valence.loc[w]['val'] for w in pw_valence]
        pv_arousal = [lexdf_arousal.loc[w]['val'] for w in pw_arousal]
        pv_dominance = [lexdf_dominance.loc[w]['val'] for w in pw_dominance]

        numTokens = len(at)
        numLexTokens_valence = len(pw_valence)
        numLexTokens_arousal = len(pw_arousal)
        numLexTokens_dominance = len(pw_dominance)

        avgValence = np.mean(pv_valence) if numLexTokens_valence > 0 else np.nan
        avgArousal = np.mean(pv_arousal) if numLexTokens_arousal > 0 else np.nan
        avgDominance = np.mean(pv_dominance) if numLexTokens_dominance > 0 else np.nan

        return [numTokens, numLexTokens_valence, avgValence, numLexTokens_arousal, avgArousal, numLexTokens_dominance, avgDominance]


def process_df(df, lexdf_valence, lexdf_arousal, lexdf_dominance):
        logging.info("Number of rows: " + str(len(df)))
        print("process_df started")  # 确认进入了 process_df

        # 为每条文本分别计算情感分值
        resrows = [get_vals(x, lexdf_valence, lexdf_arousal, lexdf_dominance) for x in df['text']]
        print(f"Processed {len(resrows)} rows")  # 打印处理的行数
        resrows = [x + y for x, y in zip(df.values.tolist(), resrows)]

        resdf = pd.DataFrame(resrows, columns=df.columns.tolist() + ['numTokens', 'numLexTokens_valence', 'avgValence',
                                                                     'numLexTokens_arousal', 'avgArousal',
                                                                     'numLexTokens_dominance', 'avgDominance'])

        print(resdf.head())  # 打印输出的前几行，确认数据被正确处理

        return resdf

def main(dataPath, lexPath, savePath):
    print("Main function started")  # 调试输出，确保进入了 main 函数
    os.makedirs(savePath, exist_ok=True)

    logfile = os.path.join(savePath, 'log.txt')
    logging.basicConfig(filename=logfile, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

    df = pd.read_csv(dataPath)
    print(f"Data loaded with {len(df)} rows")  # 调试输出，确认数据被正确加载

    # 分别加载 valence, arousal 和 dominance 的词典
    lexdf_valence = prep_dim_lexicon(read_lexicon(lexPath, ['valence']), 'valence')
    lexdf_arousal = prep_dim_lexicon(read_lexicon(lexPath, ['arousal']), 'arousal')
    lexdf_dominance = prep_dim_lexicon(read_lexicon(lexPath, ['dominance']), 'dominance')

    print("Lexicons loaded")  # 调试输出，确认词典被加载
    resdf = process_df(df, lexdf_valence, lexdf_arousal, lexdf_dominance)

    # 获取输入文件名，并加上 'vad' 后缀作为输出文件名
    base_name = os.path.basename(dataPath)  # 获取文件名，例如 'cn.csv'
    file_name, _ = os.path.splitext(base_name)  # 去掉文件扩展名，得到 'cn'
    output_file = f"{file_name}_vad.csv"  # 拼接 '_vad'，得到 'cn_vad.csv'

    print(f"Saving results to {output_file}")  # 调试输出，确认输出文件名
    resdf.to_csv(os.path.join(savePath, output_file), index=False)
    print("Results saved")  # 确认保存结果


if __name__ == '__main__':
        args = parser.parse_args()

        dataPath = args.dataPath
        lexPath = args.lexPath
        savePath = args.savePath

        # 调用 main 函数时，传递正确的参数
        main(dataPath, lexPath, savePath)
