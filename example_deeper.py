import math
from certa.eval import expl_eval, mean_drop, mean_impact
from certa.triangles_method import explainSamples
from certa.local_explain import dataset_local
from certa.local_explain import find_similarities
from keras.models import load_model
import pandas as pd
import numpy as np
import os
import gensim.downloader as api
import models.DeepER as dp


def merge_sources(table, left_prefix, right_prefix, left_source, right_source, copy_from_table, ignore_from_table):
    dataset = pd.DataFrame(
        columns={col: table[col].dtype for col in copy_from_table})
    ignore_column = copy_from_table + ignore_from_table

    for _, row in table.iterrows():
        leftid = row[left_prefix + 'id']
        rightid = row[right_prefix + 'id']

        new_row = {column: row[column] for column in copy_from_table}

        try:
            for id, source, prefix in [(leftid, left_source, left_prefix), (rightid, right_source, right_prefix)]:

                for column in source.keys():
                    if column not in ignore_column:
                        new_row[prefix + column] = source.loc[id][column]

            dataset = dataset.append(new_row, ignore_index=True)
        except:
            pass
    return dataset


def to_deeper_data(df: pd.DataFrame):
    res = []
    for r in range(len(df)):
        row = df.iloc[r]
        lpd = row.filter(regex='^ltable_')
        rpd = row.filter(regex='^rtable_')
        if 'label' in row:
            label = row['label']
            res.append((lpd.values.astype('str'),
                        rpd.values.astype('str'), label))
        else:
            res.append((lpd.values.astype('str'), rpd.values.astype('str')))
    return res


def predict_fn(x, m, ignore_columns=['ltable_id', 'rtable_id', 'label']):
    data = to_deeper_data(
        x.drop([c for c in ignore_columns if c in x.columns], axis=1))
    out = dp.predict(data, model, embeddings_model, tokenizer)
    out_df = pd.DataFrame(out, columns=['nomatch_score', 'match_score'])
    out_df.index = x.index
    return pd.concat([x.copy(), out_df], axis=1)


def get_original_prediction(r1, r2):
    lprefix = 'ltable_'
    rprefix = 'rtable_'
    r1_df = pd.DataFrame(data=[r1.values], columns=r1.index)
    r2_df = pd.DataFrame(data=[r2.values], columns=r2.index)
    r1_df.columns = list(map(lambda col: lprefix + col, r1_df.columns))
    r2_df.columns = list(map(lambda col: rprefix + col, r2_df.columns))
    r1r2 = pd.concat([r1_df, r2_df], axis=1)
    r1r2['id'] = "0@" + str(r1r2[lprefix + 'id'].values[0]) + \
        "#" + "1@" + str(r1r2[rprefix + 'id'].values[0])
    r1r2 = r1r2.drop([lprefix + 'id', rprefix + 'id'], axis=1)
    return predict_fn(r1r2, model)[['nomatch_score', 'match_score']].values[0]


datadir = 'datasets/beers/'
lsource = pd.read_csv(datadir + 'tableA.csv')
rsource = pd.read_csv(datadir + 'tableB.csv')
gt = pd.read_csv(datadir + 'train.csv')
valid = pd.read_csv(datadir + 'valid.csv')
test = pd.read_csv(datadir + 'test.csv')

train_df = merge_sources(gt, 'ltable_', 'rtable_',
                         lsource, rsource, ['label'], ['id'])
valid_df = merge_sources(valid, 'ltable_', 'rtable_',
                         lsource, rsource, ['label'], ['id'])
test_df = merge_sources(test, 'ltable_', 'rtable_',
                        lsource, rsource, ['label'], ['id'])

if not os.path.exists('models/glove.6B.50d.txt'):
    word_vectors = api.load("glove-wiki-gigaword-50")
    word_vectors.save_word2vec_format('models/glove.6B.50d.txt', binary=False)

embeddings_index = dp.init_embeddings_index('models/glove.6B.50d.txt')
emb_dim = len(embeddings_index['cat'])
embeddings_model, tokenizer = dp.init_embeddings_model(embeddings_index)


if os.path.exists("DeepER_best_model.h5"):
    model = load_model("DeepER_best_model.h5")
else:
    model = dp.init_DeepER_model(emb_dim)
    model = dp.train_model_ER(to_deeper_data(
        train_df), model, embeddings_model, tokenizer)


theta_min, theta_max = find_similarities(test_df, False)

stringa_vuota = []
attributi_random = []

for nt in [int(math.log(min(len(lsource), len(rsource)))), 10, 50]:
    print('running CERTA with nt='+str(nt))
    for i in range(1, 101):
        l_tuple = lsource.iloc[i]
        r_tuple = rsource.iloc[i]
        local_samples = dataset_local(l_tuple, r_tuple, model, lsource, rsource, datadir, theta_min, theta_max, predict_fn,
                                      num_triangles=nt)

        prediction = get_original_prediction(l_tuple, r_tuple)
        class_to_explain = np.argmax(prediction)

        explanation, flipped_pred = explainSamples(local_samples, [lsource, rsource], model, predict_fn,
                                                   class_to_explain=class_to_explain, maxLenAttributeSet=3)
        print(explanation)

        for exp in explanation:
            e_attrs = exp.split('/')
            e_score = explanation[exp]
            expl_evaluation = expl_eval(class_to_explain, e_attrs, e_score, lsource, l_tuple, model, prediction, rsource,
                                        r_tuple, predict_fn)
            # Mi salvo le tuple modificate con stringa vuota
            stringa_vuota.append(expl_evaluation.loc[0, :])
            # Mi salvo le tuple modificate con attributi a caso
            attributi_random.append(expl_evaluation.loc[1, :])
            print(expl_evaluation.head())
            break
        print('------------------------------------')
    drop_medio = mean_drop(stringa_vuota, attributi_random)
    impact_medio = mean_impact(stringa_vuota, attributi_random)
    print('*********************************************')
    print('Il mean_drop con ' + str(nt) +
          ' triangoli è pari a ' + str(drop_medio))
    print('Il mean_impact con ' + str(nt) +
          ' triangoli è pari a ' + str(impact_medio))
    print('*********************************************')
