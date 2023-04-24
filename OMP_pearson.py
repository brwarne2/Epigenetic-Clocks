import pandas as pd
from src.scripts.features import judge_features
from sklearn.impute import KNNImputer
# from src.elm import nano_elm
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
from scipy.stats import pearsonr
import scipy

def gramSchmidt_modified(A):
    """
    Applies the Gram-Schmidt method to A
    and returns Q and R, so Q*R = A.
    """
    Q = np.zeros(A.shape)
    for k in range(0, 1):
        R = np.sqrt(np.dot(A[:, k], A[:, k]))
        if R == 0:
            Q[:, k] = 0
            print()
        else:
            # print(R[k, k])
            Q[:, k] = A[:, k]/R
        for j in range(k+1, A.shape[1]):
            R = np.dot(Q[:, k], A[:, j])
            # print(np.sqrt(np.dot(A[:, j], A[:, j])))
            # print(R[k, j])
            A[:, j] = A[:, j] - R*Q[:, k]

    return Q, A

def preparation(df):
    first_three_cols = pd.read_csv(
        "file_path.csv",  # insert your file path
        header=1,
    ).reset_index()
    df = df.iloc[1:, 2:]
    df = pd.concat([first_three_cols, df], axis=1)
    df = df[df["tissue"] == "whole blood"]
    df = df[df["age"] > 17]
    age = df['age'].reset_index(drop=True)
    df = df.iloc[:, 5:].reset_index(drop=True)
    return df, age

def main():

    big_df = pd.DataFrame()
    out_df = pd.DataFrame()
    # parse over methyl files to save in single large df
    for i in range(485):  # 485 total files
        i += 1
        print(i)
        df = pd.read_csv(
            "file_path.csv",  # insert your file path
            delimiter=",", header=0, index_col=0)
        if i == 1:
            df = df.iloc[1:, 3:]
        else:
            df = df.iloc[1:, :]
        df = preparation(df)[0]
        if i == 1:
            age = preparation(df)[1]
            big_df = pd.concat([age, df], axis=1)
        print('replace')
        df = df.replace(-999, np.nan)
        print('count nans')
        n_na = df.isna().sum().sum()
        if n_na > 0:
            # impute with column means
            print('filling nans')
            df = df.fillna(df.mean())
        print('concat')

        # use only top n correlated features with the output
        correlations = []
        for c in range(df.shape[1]):
            col = df.iloc[:, c]
            corr = scipy.stats.pearsonr(col, age)[0]
            correlations.append(corr)
        best_df = pd.DataFrame()
        for i in range(50):
            best_index = np.argmax(correlations)
            best_col = df.iloc[:, best_index]
            best_df[best_index] = best_col
            correlations[best_index] = 0
        out_df = pd.concat([out_df, best_df], axis=1)

    scaler = StandardScaler()
    age = big_df['age']
    big_df = pd.DataFrame(scaler.fit_transform(out_df))
    X_train = big_df.iloc[:int(big_df.shape[0]*0.8), :-1]
    X_train_norm = pd.DataFrame(scaler.fit_transform(X_train))
    Y_train = big_df.iloc[:int(big_df.shape[0]*0.8), -1]
    X_test = big_df.iloc[int(big_df.shape[0]*0.8):, :-1]
    X_test_norm = pd.DataFrame(scaler.fit_transform(X_test))
    Y_test = big_df.iloc[int(big_df.shape[0]*0.8):, -1]
    big_df_norm = big_df.iloc[:, 1:]
    # save column names
    col_names_list = list(big_df.columns)
    correlations = []
    # pearson
    for c in range(big_df_norm.shape[1]):
        col = big_df_norm.iloc[:, c]
        corr = scipy.stats.pearsonr(col, age)[0]
        correlations.append(corr)
    # new NL
    # correlations = nano_elm.judge_features(X_train_norm, Y_train, X_test_norm, Y_test)
    # old NL
    # correlations = nano_elm.judge_features(big_df_norm, y, regression=True)

    # set nans equal to zero
    correlations = [0 if x != x else x for x in correlations]
    best_features_df = pd.DataFrame()
    best_correlations = []
    best_NL_correlations = []
    best_standard_correlations = []
    norms = []
    column_names_list = []
    # loop the amount of times as there are columns
    for N in range(X_train.shape[1]):
        print(N)
        # for n in range(big_df_norm.shape[1]):  # looping over columns
        #     X = big_df_norm.iloc[:, n]  # select the nth column
        #     # non-linear correlation
        #     cor = nano_elm.judge_features(X_train, Y_train, X_test, Y_test)[0]
        #     abs_cor = abs(float(cor))
        #     correlations.append(abs_cor)
        #     col_name = col_names_list[n]
        #     local_col_names.append(col_name)
            # # standard correlation
            # X = big_df_norm_standard.iloc[:, n]
            # standard_cor = abs(X.corr(y))
            # if math.isnan(standard_cor) == True:
            #     standard_cor = 0
            # standard_correlations.append(standard_cor)
        # # find column with highest non-linear correlation
        # best_col_index = np.argmax(correlations)
        # best_correlation = correlations[best_col_index]
        # best_correlations.append(best_correlation)
        # best_col = big_df_norm.iloc[:, best_col_index]
        # best_col_name = col_names_list[best_col_index]
        # # s_cor_best_col = abs(best_col.corr(age))
        # s_corr_best_col, _ = pearsonr(best_col, y)
        # print('Pearsons correlation of best column according to non-linear correlation: %.3f' % s_corr_best_col)
        # print('best non-linear correlation: ' + str(best_correlation))
        # print('best column: ' + str(best_col_name))

        # standard correlation
        best_col_index = np.argmax(correlations)
        best_col = big_df_norm.iloc[:, best_col_index]
        best_correlation = correlations[best_col_index]
        print('pearson correlation:')
        print(best_correlation)
        best_correlations.append(best_correlation)
        correlations.pop(best_col_index)
        # best_standard_correlation = abs(best_s_col.corr(age))
        # best_correlation, _ = pearsonr(best_col, age)
        # print('Pearsons correlation of best column according to NL: %.3f' % best_correlation)
        # best_standard_correlations.append(best_correlation)
        best_col_name = best_col.name
        column_names_list.append(best_col_name)
        # print('best standard correlation: ' + str(best_standard_correlation))
        print('column name: ' + str(best_col_name))
        print('-----------------------------------------')

        # save column and drop it
        best_features_df[N] = best_col
        best_features_df = best_features_df.rename(columns={N: best_col_name})
        big_df_norm = big_df_norm.drop(best_col_name, axis=1)
        # orthogonalize that feature with respect to all of the rest of the features
        big_df_norm = pd.concat([best_col, big_df_norm], axis=1)
        A_matrix = big_df_norm.to_numpy()
        Q, A = gramSchmidt_modified(A_matrix)
        A = A[:, 1:]
        norm = math.sqrt(np.dot(best_col, best_col))
        norms.append(norm)
        col_names = big_df_norm.columns[1:]
        big_df_norm = pd.DataFrame(A, columns=col_names)

        # # save column and drop it (pearson)
        # best_features_df[N] = best_s_col
        # best_features_df = best_features_df.rename(columns={N: best_s_col_name})
        # big_df_norm_standard = big_df_norm_standard.drop(best_s_col_name, axis=1)
        # # orthogonalize that feature with respect to all of the rest of the features
        # big_df_norm_standard = pd.concat([best_s_col, big_df_norm_standard], axis=1)
        # A_matrix = big_df_norm_standard.to_numpy()
        # Q, A = gramSchmidt_modified(A_matrix)
        # A = A[:, 1:]
        # norm = math.sqrt(np.dot(best_s_col, best_s_col))
        # norms.append(norm)
        # col_names = big_df_norm_standard.columns[1:]
        # big_df_norm_standard = pd.DataFrame(A, columns=col_names)
        if float(norm) < float(1):
            print()

        if best_features_df.shape[1] == 50:
            best_features_df_50 = pd.concat([best_features_df, age], axis=1)
            best_features_df_50.to_csv('MP_methyl_best_features_50.csv')

        if best_features_df.shape[1] == 100:
            best_features_df_100 = pd.concat([best_features_df, age], axis=1)
            best_features_df_100.to_csv('MP_methyl_best_features_100.csv')

        if best_features_df.shape[1] == 200:
            best_features_df_200 = pd.concat([best_features_df, age], axis=1)
            best_features_df_200.to_csv('MP_methyl_best_features_200.csv')

        if best_features_df.shape[1] == 300:
            best_features_df_300 = pd.concat([best_features_df, age], axis=1)
            best_features_df_300.to_csv('MP_methyl_best_features_300.csv')

        if best_features_df.shape[1] == 500:
            best_features_df_500 = pd.concat([best_features_df, age], axis=1)
            best_features_df_500.to_csv('MP_methyl_best_features_500.csv')

        if best_features_df.shape[1] == 700:
            best_features_df_700 = pd.concat([best_features_df, age], axis=1)
            best_features_df_700.to_csv('MP_methyl_best_features_700.csv')

        if best_features_df.shape[1] == 1000:
            best_features_df_1000 = pd.concat([best_features_df, age], axis=1)
            best_features_df_1000.to_csv('MP_methyl_best_features_1000.csv')

        if best_features_df.shape[1] == 1300:
            best_features_df_1300 = pd.concat([best_features_df, age], axis=1)
            best_features_df_1300.to_csv('MP_methyl_best_features_1300.csv')
        if best_features_df.shape[1] == best_features_df.shape[0]:
            pd.DataFrame(norms).to_csv('methyl_norms_NL.csv')
            # pd.DataFrame(best_correlations).to_csv('NHANES_best_corrs_pearson.csv')
            best_features_df = pd.concat([best_features_df, age], axis=1)
            best_features_df.to_csv('MP_methyl_best_features_NL.csv')
            exit()
        # # save column and drop it (pearson)
        # best_features_df[N] = best_s_col
        # best_features_df = best_features_df.rename(columns={N: best_s_col_name})
        # big_df_norm_standard = big_df_norm_standard.drop(best_s_col_name, axis=1)
        # # orthogonalize that feature with respect to all of the rest of the features
        # big_df_norm_standard = pd.concat([best_s_col, big_df_norm_standard], axis=1)
        # A_matrix = big_df_norm_standard.to_numpy()
        # Q, A = gramSchmidt_modified(A_matrix)
        # A = A[:, 1:]
        # norm = math.sqrt(np.dot(best_s_col, best_s_col))
        # norms.append(norm)
        # col_names = big_df_norm_standard.columns[1:]
        # big_df_norm_standard = pd.DataFrame(A, columns=col_names)

    pd.DataFrame(norms).to_csv('methyl_norms_NL.csv')
    # pd.DataFrame(best_correlations).to_csv('NHANES_best_corrs_pearson.csv')
    best_features_df = pd.concat([best_features_df, age], axis=1)
    best_features_df.to_csv('MP_methyl_best_features_NL_new.csv')
    # age.to_csv('NHANES_age.csv')
    # pearson_df = big_df[column_names_list].reset_index(drop=True)
    # pearson_df = pd.concat([pearson_df, age], axis=1)
    # pearson_df.to_csv('NHANES_226_pearson.csv')
    pd.DataFrame(best_correlations).to_csv('methyl_best_pearsons_NL_new.csv')
    pd.DataFrame(best_features_df.columns).to_csv('methyl_columns_NL_new.csv')
    print()


if __name__ == '__main__':
    main()