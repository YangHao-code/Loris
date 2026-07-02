from snorkel.labeling import PandasLFApplier, filter_unlabeled_dataframe
from snorkel.labeling.model import LabelModel
import sys
import os
import numpy as np 
import pandas as pd
import re
import pickle
# from wrench.labelmodel import DawidSkene, FlyingSquid, MeTaL, HyperLM, EBCC, IBCC, Fable
# from wrench.labelmodel.majority_voting import MajorityVoting, MajorityWeightedVoting

sys.path.append(os.path.join(os.getcwd(), ".."))


def clean_text(text):
    text = text.encode("ascii", "ignore").decode()
    text = re.sub('[^a-zA-Z0-9\s\.\,\!\?\;\:\'\"]', ' ', text)  # Allowing common punctuation
    # punctuationfree="".join([i for i in text if i not in text.lower()])
    res = text.lower()

    # there are some corner cases where the processed text is too short or empty
    # use this part to retur None so we can filter them out
    if(len(res.strip().split(' '))<=2):
        return None
    return text.lower()


# def run_snorkel_with_funcs(dataset_name, funcs, conn, cardinality):
    
#     sentences_df=pd.read_sql(f'SELECT * FROM {dataset_name}', conn)
#     sentences_df = sentences_df.rename(columns={"class": "expected_label", "content": "old_text"})
#     sentences_df['text'] = sentences_df['old_text'].apply(lambda s: clean_text(s))
#     sentences_df = sentences_df[~sentences_df['text'].isna()]
#     applier = PandasLFApplier(lfs=funcs)
#     initial_vectors = applier.apply(df=sentences_df, progress_bar=False)
#     print(f"initial_vectors: {initial_vectors.shape}")
#     print(f"initial_vectors:\n {initial_vectors}")
#     with open('initial_vectors.pkl', 'wb') as f:
#         pickle.dump(initial_vectors, f)
    
#     model = LabelModel(cardinality=cardinality, verbose=True, device='cpu')
#     model.fit(L_train=initial_vectors, n_epochs=500, log_freq=100, seed=123)
#     probs_test= model.predict_proba(L=initial_vectors)
#     df_sentences_filtered, probs_test_filtered, filtered_vectors, df_no_signal  = filter_unlabeled_dataframe(
#         X=sentences_df, y=probs_test, L=initial_vectors
#     )	

#     df_sentences_filtered = df_sentences_filtered.reset_index(drop=True)
#     prob_diffs = [abs(t[0]-t[1]) for t in probs_test_filtered]
#     prob_diffs_tuples = [(t[0],t[1]) for t in probs_test_filtered]
#     df_sentences_filtered['model_pred_diff'] = pd.Series(prob_diffs)
#     df_sentences_filtered['model_pred_prob_tuple'] = pd.Series(prob_diffs_tuples)
#     df_sentences_filtered['model_pred'] = pd.Series(model.predict(L=filtered_vectors))

#     wrong_preds = df_sentences_filtered[(df_sentences_filtered['expected_label']!=df_sentences_filtered['model_pred'])]
#     # df_sentences_filtered.to_csv('predictions_shakira.csv', index=False)
#     # logger.critical(wrong_preds)
#     global_accuray_on_valid=(len(df_sentences_filtered)-len(wrong_preds))/len(df_sentences_filtered)

#     print(f"""
#         out of {len(sentences_df)} sentences, {len(df_sentences_filtered)} actually got at least one signal to \n
#         make prediction. Out of all the valid predictions, we have {len(wrong_preds)} wrong predictions, \n
#         accuracy = {(len(df_sentences_filtered)-len(wrong_preds))/len(df_sentences_filtered)} 
#     """)
    
#     global_accuracy = (len(df_sentences_filtered)-len(wrong_preds))/len(sentences_df)
    
    
#     ground_truth = df_sentences_filtered['expected_label']
#     snorkel_predictions = df_sentences_filtered['model_pred']
#     snorkel_probs = df_sentences_filtered['model_pred_diff']
#     df_sentences_filtered['vectors'] = pd.Series([",".join(map(str, t)) for t in filtered_vectors])
#     correct_predictions = (snorkel_predictions == ground_truth)
#     incorrect_predictions = (snorkel_predictions != ground_truth)
#     correct_preds_by_snorkel = df_sentences_filtered[correct_predictions].reset_index(drop=True)
#     wrong_preds_by_snorkel = df_sentences_filtered[incorrect_predictions].reset_index(drop=True)
    
#     return df_sentences_filtered, correct_preds_by_snorkel, wrong_preds_by_snorkel, filtered_vectors, correct_predictions, \
#           incorrect_predictions, global_accuracy, global_accuray_on_valid 

def run_label_model_with_funcs(dataset_name, funcs, conn, cardinality, model_type="snorkel"):
    # Load and clean data
    sentences_df = pd.read_sql(f'SELECT * FROM {dataset_name}', conn)
    sentences_df = sentences_df.rename(columns={"class": "expected_label", "content": "old_text"})
    sentences_df['text'] = sentences_df['old_text'].apply(lambda s: clean_text(s))
    sentences_df = sentences_df[~sentences_df['text'].isna()]

    # Apply labeling functions
    applier = PandasLFApplier(lfs=funcs)
    initial_vectors = applier.apply(df=sentences_df, progress_bar=False)
    print(f"initial_vectors: {initial_vectors.shape}")
    with open('initial_vectors.pkl', 'wb') as f:
        pickle.dump(initial_vectors, f)

    # Fit model
    if model_type == "snorkel":
        model = LabelModel(cardinality=cardinality, verbose=True, device='cpu')
        model.fit(L_train=initial_vectors, n_epochs=500, log_freq=100, seed=123)
        probs_test = model.predict_proba(L=initial_vectors)
        predictions = model.predict(L=initial_vectors)

    elif model_type == "dawidskene":
        model = DawidSkene(n_epochs=100, tolerance=1e-4)
        model.fit(initial_vectors, n_class=cardinality)
        probs_test = model.predict_proba(initial_vectors)
        predictions = np.argmax(probs_test, axis=1)

    elif model_type == "flyingsquid":
        model = FlyingSquid()
        model.fit(initial_vectors, n_class=cardinality)
        probs_test = model.predict_proba(initial_vectors)
        predictions = np.argmax(probs_test, axis=1)

    elif model_type == "metal":
        model = MeTaL()
        model.fit(initial_vectors, n_class=cardinality)
        probs_test = model.predict_proba(initial_vectors)
        predictions = np.argmax(probs_test, axis=1)
    
    elif model_type == "hyperlm":
        model = HyperLM()
        model.fit(initial_vectors, n_class=cardinality)
        probs_test = model.predict_proba(initial_vectors)
        predictions = np.argmax(probs_test, axis=1)
    elif model_type == "majority":
        model = MajorityVoting()
        model.fit(initial_vectors, n_class=cardinality)
        probs_test = model.predict_proba(initial_vectors)
        predictions = np.argmax(probs_test, axis=1)
    elif model_type == "weighted_majority":
        model = MajorityWeightedVoting()
        model.fit(initial_vectors, n_class=cardinality)
        probs_test = model.predict_proba(initial_vectors)
        predictions = np.argmax(probs_test, axis=1)
    elif model_type == "ebcc":
        model = EBCC()
        model.fit(initial_vectors, n_class=cardinality)
        probs_test = model.predict_proba(initial_vectors)
        predictions = np.argmax(probs_test, axis=1)
    elif model_type == "ibcc":
        model = IBCC()
        model.fit(initial_vectors, n_class=cardinality)
        probs_test = model.predict_proba(initial_vectors)
        predictions = np.argmax(probs_test, axis=1)
    elif model_type == "fable":
        model = Fable()
        model.fit(initial_vectors, n_class=cardinality)
        probs_test = model.predict_proba(initial_vectors)
        predictions = np.argmax(probs_test, axis=1)


    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Filter unlabeled
    df_sentences_filtered, probs_test_filtered, filtered_vectors, df_no_signal = filter_unlabeled_dataframe(
        X=sentences_df, y=probs_test, L=initial_vectors
    )

    df_sentences_filtered = df_sentences_filtered.reset_index(drop=True)
    prob_diffs = [abs(t[0] - t[1]) for t in probs_test_filtered]
    prob_diffs_tuples = [tuple(t) for t in probs_test_filtered]
    df_sentences_filtered['model_pred_diff'] = pd.Series(prob_diffs)
    df_sentences_filtered['model_pred_prob_tuple'] = pd.Series(prob_diffs_tuples)
    df_sentences_filtered['model_pred'] = pd.Series(np.argmax(probs_test_filtered, axis=1))

    # Accuracy
    wrong_preds = df_sentences_filtered[
        df_sentences_filtered['expected_label'] != df_sentences_filtered['model_pred']
    ]
    global_accuracy = (len(df_sentences_filtered) - len(wrong_preds)) / len(sentences_df)
    global_accuracy_on_valid = (len(df_sentences_filtered) - len(wrong_preds)) / len(df_sentences_filtered)

    print(f"""
        Out of {len(sentences_df)} sentences, {len(df_sentences_filtered)} got signal for prediction.
        {len(wrong_preds)} predictions were incorrect.
        Accuracy on valid = {global_accuracy_on_valid:.4f}
        Overall accuracy = {global_accuracy:.4f}
    """)

    # Return
    correct_predictions = df_sentences_filtered['expected_label'] == df_sentences_filtered['model_pred']
    incorrect_predictions = ~correct_predictions
    correct_preds_by_model = df_sentences_filtered[correct_predictions].reset_index(drop=True)
    wrong_preds_by_model = df_sentences_filtered[incorrect_predictions].reset_index(drop=True)

    return (
        df_sentences_filtered,
        correct_preds_by_model,
        wrong_preds_by_model,
        filtered_vectors,
        correct_predictions,
        incorrect_predictions,
        global_accuracy,
        global_accuracy_on_valid,
    )

def select_user_input(user_confirm_size,
                     user_complaint_size,
                     random_state,
                     filtered_vectors,
                     correct_preds_by_snorkel,
                     wrong_preds_by_snorkel,
                      correct_predictions,
                      incorrect_predictions ):

    user_confirm_df = correct_preds_by_snorkel.sample(n=user_confirm_size, random_state=random_state)
    user_complaints_df = wrong_preds_by_snorkel.sample(n=user_complaint_size, random_state=random_state)
    
    random_confirm_indices = user_confirm_df.index
    random_complaints_indices = user_complaints_df.index
    random_user_confirms_vecs = filtered_vectors[correct_predictions][random_confirm_indices]
    random_user_complaints_vecs = filtered_vectors[incorrect_predictions][random_complaints_indices]
    user_input_df = pd.concat([user_confirm_df, user_complaints_df])
    gts = user_input_df['expected_label'].reset_index(drop=True)
    user_vecs = np.vstack((random_user_confirms_vecs, random_user_complaints_vecs))
    
    return user_vecs, gts, user_input_df


def construct_input_df_to_solver(user_vecs, gts):
    df_user_vectors = pd.DataFrame(user_vecs, columns=[f'lf_{i+1}' for i in range(user_vecs.shape[1])])
    combined_df= pd.concat([df_user_vectors, gts], axis=1)
    
    return combined_df

def create_solver_input_df_copies(lf_names_after_fix, user_input_df, res_df):
    df_copies = {}

    cols_needed = ['text', 'expected_label', 'cid']

    # Loop through each column in df2 and create a copy of df1 with modified 'expected_label'
    for lf in lf_names_after_fix:
        # Create a deep copy of df1
        df_copy = user_input_df.copy(deep=True)

        # Update the 'expected_label' column based on the corresponding column in df2
        df_copy['expected_label'] = res_df[lf].values

        # Store the modified dataframe in the dictionary with key as the labeling function name
        df_copies[lf] = df_copy[cols_needed]
    
    return df_copies

