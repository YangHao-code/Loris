import pandas as pd
import itertools

def get_data_analytics(foldername, file):
    trainpath = foldername + '/' + file + '_train.json'
    train_df = pd.read_json(trainpath, lines=True)
    testpath = foldername + '/' + file + '_test.json'
    test_df = pd.read_json(testpath, lines=True)
    df_all = pd.concat([train_df,test_df])
    label_li = [x for x in df_all['doc_label'].values]
    labels = list(itertools.chain.from_iterable(label_li))
    print("train samples:", len(train_df))
    print("test samples:", len(test_df))
    print("labels:", len(set(labels)))
    cardinality = sum([len(x) for x in df_all['doc_label'].values])/len(df_all)
    print("cardinality:", round(cardinality,3))
    print("density:", round(cardinality/len(set(labels)),3))
    token_li = [x for x in df_all['doc_token'].values]
    tokens = list(itertools.chain.from_iterable(token_li))
    print("vocabulary:", len(set(tokens)))
    return train_df, test_df, set(labels)
# foldername = "medical"
# trainpath = foldername + '/' + foldername + '_train.json'
# train_df = pd.read_json(trainpath, lines=True)
# testpath = foldername + '/' + foldername + '_test.json'
# test_df = pd.read_json(testpath, lines=True)
# df_all = pd.concat([train_df,test_df])
# label_li = [x for x in df_all['doc_label'].values]
# labels = list(itertools.chain.from_iterable(label_li))
# print("train samples:", len(train_df))
# print("test samples:", len(test_df))
# print("labels:", len(set(labels)))
# print("cardinality:", sum([len(x) for x in df_all['doc_label'].values])/len(df_all))

if __name__ == '__main__':
    file = "bibtex"
    foldername = "../data/" + file

    train_df, test_df, labels = get_data_analytics(foldername, file)
    labelratio = []
    for x in labels:
        labelratio.append(train_df['doc_label'].apply(lambda y: x in y).sum()/len(train_df))
    df = pd.DataFrame({"label": list(labels), "density": labelratio})
    df.sort_values(by="density", ascending=False, inplace=True)
    df = df[0:10]
    print(df)
    # print(df.to_latex(index=False))