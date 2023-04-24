import pandas as pd
from sklearn.decomposition import PCA
import scipy
import numpy

def preparation(df):
    first_three_cols = pd.read_csv(
        "file_name.csv",  # insert your file path
        header=1,
    ).reset_index()
    df = df.iloc[1:, 2:]
    df = pd.concat([first_three_cols, df], axis=1)
    df = df[df["tissue"] == "whole blood"]
    df = df[df["age"] > 17]
    age = df['age'].reset_index(drop=True)
    df = df.iloc[:, 5:].reset_index(drop=True)
    # perc = 15.0  # Like N %
    # min_count = int(((100 - perc) / 100) * df.shape[0] + 1)
    # clean_df = df.dropna(axis=1, thresh=min_count)
    return df, age

def main():
    out_df = pd.DataFrame()
    # parse over methyl files to save in single large df
    for i in range(485):  # 485 total files
        i += 1
        print(i)
        df = pd.read_csv(
            "file_name.csv"  # insert your file path
            , delimiter=",", header=0, index_col=0)
        if i == 1:
            df = df.iloc[1:, 3:]
        else:
            df = df.iloc[1:, :]
        df = preparation(df)[0]
        if i == 1:
            age = preparation(df)[1]
            # big_df = pd.concat([age, df], axis=1)
        print('replace')
        df = df.replace(-999, np.nan)
        # print('count nans')
        n_na = df.isna().sum().sum()
        if n_na > 0:
            # impute with column means
            # print('filling nans')
            df = df.fillna(df.mean())
        # print('concat')
        print(df.shape)
        # pearson
        correlations = []
        for c in range(df.shape[1]):
            col = df.iloc[:, c]
            corr = scipy.stats.pearsonr(col, age)[0]
            correlations.append(corr)
        best_df = pd.DataFrame()
        for i in range(20):
            best_index = np.argmax(correlations)
            best_col = df.iloc[:, best_index]
            best_df[best_index] = best_col
            correlations[best_index] = 0
        pca = PCA(n_components=3)
        pca_df = pd.DataFrame(pca.fit_transform(best_df))
        explained_variance = pca.explained_variance_ratio_
        print(explained_variance)
        out_df = pd.concat([out_df, pca_df], axis=1)
    age = age.reset_index(drop=True)
    out_df = out_df.reset_index(drop=True)
    out_df = pd.concat([out_df, age], axis=1)
    out_df.to_csv('methyl_PCA.csv')

main()