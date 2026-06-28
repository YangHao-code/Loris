import pdb
import random
from sklearn.manifold import TSNE
import numpy as np
from scipy.spatial.distance import cdist
import torch
from sklearn.cluster import KMeans
from torch.nn.functional import normalize
from torch.nn.functional import one_hot
import math
from datetime import datetime
from libact.models.multilabel import BinaryRelevance, DummyClf
from sklearn.svm import SVC
from libact.models import LogisticRegression, SklearnProbaAdapter
from libact.base.dataset import Dataset
from libact.utils import seed_random_state
import pdb
from joblib import Parallel, delayed
from libact.query_strategies.multilabel.cost_sensitive_reference_pair_encoding import CSRPE
from libact.utils.multilabel import pairwise_f1_score
from sklearn.metrics.pairwise import paired_distances
from scipy.spatial.distance import hamming
from audi import LabelRankingModel
from scipy.spatial import distance
from scipy.stats import entropy as scipy_entropy
from gpb2m import B2M
from sklearn.gaussian_process.kernels import RBF
from itertools import combinations
from math import comb
import os, sys
from config import Config


def GPB2M(conf, logits_E_X_Y, batch_size, unlabeled_indices, cur_acqIdx):
    labeled_pool = conf.al_info.train_logit_save.train_logits
    labeled_pool = torch.stack(labeled_pool).cpu().numpy()
    X_pool = torch.mean(logits_E_X_Y.cpu(),0).numpy()
    Y = conf.al_info.train_logit_save.train_standard_labels
    C = np.zeros(labeled_pool.shape)
    for i in range(labeled_pool.shape[0]):
        C[i, Y[i]] = 1
    Y = C
    b2mModel=B2M()
    b2mModel.fit(labeled_pool, Y, K=10,kernel_function=RBF(length_scale=1),K_threds=0.75)
    b2mModel.learn(learnIter=15)
    sampleRes=b2mModel.ALsample(X_pool,eta=0,test=False,sampleType='en',testX=None,testY=None)
    # pdb.set_trace()
    topkIdx = sampleRes[0:batch_size]
    return np.array(unlabeled_indices)[topkIdx].tolist()


## reference paper
## Effective active learning strategy for multi-label learning
## https://reader.elsevier.com/reader/sd/pii/S0925231217313371?token=1E4A01DAF004BA0562C939B3C1E681A354687BA3A38C2612658F5DFE0E89ADC8C166E52271C51B1E0D25FB63150E6E7C&originRegion=us-east-1&originCreation=20230224213357

def CVIRS(conf, logits_E_X_Y, batch_size, unlabeled_indices, cur_acqIdx):

    def get_v_score(a, d, b, c, wu, su, ws, ss, Z, i):
        base = 2
        H_u = scipy_entropy([wu/q, su/q], base=base)
        H_s = scipy_entropy([ws/q, ss/q], base=base)
        tol = 1e-16
        ad = np.array((a + d), float)
        ad[np.where(ad == 0.0)[0],] = tol
        bc = np.array((b + c), float)
        bc[np.where(bc == 0.0)[0],] = tol
        ## the warning will show if the both arguments are zero due to the divisors 
        ## are zero inside scipy_entropy function, we could ignore it 
        ## we use nan to zero if the denominator is zero
        H_bcqadq = scipy_entropy([bc/q, ad/q], base=base)
        H_bcqadq = np.nan_to_num(H_bcqadq)
        H_aaddad = scipy_entropy([a/ad, d/ad], base=base)
        H_aaddad = np.nan_to_num(H_aaddad)
        H_bbccbc = scipy_entropy([b/bc, c/bc], base=base)
        H_bbccbc = np.nan_to_num(H_bbccbc)
        H_u_s = H_bcqadq + ad/q * H_aaddad + bc/q * H_bbccbc
        dist_E = (2 * H_u_s - H_u - H_s) / H_u_s
        dist_E = dist_E.squeeze() * np.array((Z < 1), dtype=float) + np.array((Z == 1), dtype=float)
        v_score = dist_E.sum() / dist_E.shape[0]
        return v_score

    X_pool = torch.mean(logits_E_X_Y.cpu(),0).numpy()
    probs_X_Y = torch.sigmoid(torch.mean(logits_E_X_Y, 0))
    # probs_X_Y1_1 = probs_X_Y.unsqueeze(-1)
    margin_X = (1 - probs_X_Y - probs_X_Y).cpu().numpy()
    margin_X = np.abs(margin_X)
    tau_X = np.argsort(np.argsort(margin_X, axis=0),axis=0)
    ## index indicates the position of margin, smaller margin has smaller index
    s_score = (tau_X.shape[0] - tau_X).sum(1)/(tau_X.shape[0] - 1)


    labeled_pool = conf.al_info.train_logit_save.train_logits
    labeled_pool = torch.stack(labeled_pool).cpu().numpy()
    Ys = conf.al_info.train_logit_save.train_standard_labels
    C = np.zeros(labeled_pool.shape)
    for i in range(labeled_pool.shape[0]):
        C[i, Ys[i]] = 1
    Ys = C
    Yu = (probs_X_Y > 0.5).int().cpu().numpy()

    ## normalised Hamming distance
    v_score_li = []
    for i in range(Yu.shape[0]):    
        Yu_i = np.expand_dims(Yu[i,:], axis=0).repeat(Ys.shape[0],0)
        a = np.sum(np.array((Yu_i + Ys) == 2, dtype=int),1, keepdims=True)
        d = np.sum(np.array((Yu_i + Ys) == 0, dtype=int),1, keepdims=True)
        b = np.sum(np.array((Yu_i - Ys) == 1, dtype=int),1, keepdims=True)
        c = np.sum(np.array((Yu_i - Ys) == -1, dtype=int),1, keepdims=True)
        q = Yu_i.shape[1]
        wu_i = np.sum(np.array(Yu_i == 1, dtype=int),1, keepdims=True)
        su_i = q - wu_i
        ws = np.sum(np.array(Ys == 1, dtype=int),1, keepdims=True)
        ss = q - ws
        Z = paired_distances(Yu_i, Ys, metric=hamming)
        dist_E = get_v_score(a, d, b, c, wu_i, su_i, ws, ss, Z, i)
        v_score_li.append(dist_E)

  
    score = s_score * np.array(v_score_li)
    topkIdx = torch.topk(torch.tensor(score), batch_size).indices.cpu().numpy().tolist()
    # ask_id = seed_random_state(conf.al_info.exp_seed).choice(np.where(score == np.max(score))[0])
    return np.array(unlabeled_indices)[topkIdx].tolist()



## reference codelink 
## https://github.com/NUAA-AL/ALiPy/blob/1b2ee2e5acc2e8651fc64759aae332853ad9e437/alipy/query_strategy/multi_label.py#L260
def audi(conf, logits_E_X_Y, batch_size, unlabeled_indices, cur_acqIdx):
    """AUDI select an instance-label pair based on Uncertainty and Diversity.
    This method will train a multilabel classification model by combining
    label ranking with threshold learning and use it to evaluate the unlabeled data.
    Thus it is no need to pass any model.

    References
    ----------
    [1] S.-J. Huang and Z.-H. Zhou. Active query driven by uncertainty and
        diversity for incremental multi-label learning. In Proceedings
        of the 13th IEEE International Conference on Data Mining, pages
        1079-1084, Dallas, TX, 2013.
    """
    epsilon = 0.5
    lr_model = LabelRankingModel()
    labeled_pool = conf.al_info.train_logit_save.train_logits
    labeled_pool = torch.stack(labeled_pool).cpu().numpy()
    X_pool = torch.mean(logits_E_X_Y.cpu(),0).numpy()
    Y = conf.al_info.train_logit_save.train_standard_labels
    C = np.zeros(labeled_pool.shape)
    for i in range(labeled_pool.shape[0]):
        C[i, Y[i]] = 1
    Y = C

    W = np.zeros(X_pool.shape)
    if int(cur_acqIdx) != 0:
        W[0: int(cur_acqIdx), :] = Y[int(conf.al_info.initial_label_size):, :]

    lr_model.fit(labeled_pool, Y)
    pres, labels = lr_model.predict(X_pool)
   
    avgP = np.mean(np.sum(Y, axis=1))
    insvals = np.abs((np.sum(labels == 1, axis=1) - avgP) / np.fmax(np.sum(W, axis=1), epsilon))
    
    topkIdx = torch.topk(torch.tensor(insvals), batch_size).indices.cpu().numpy().tolist()

    return np.array(unlabeled_indices)[topkIdx].tolist()



def CostSensitiveReferencePairEncoding(conf, logits_E_X_Y, batch_size, unlabeled_indices):
    """Cost Sensitive Reference Pair Encoding (CSRPE)
    
    References
    ----------
    .. [1] Yang, Yao-Yuan, et al. "Cost-Sensitive Reference Pair Encoding for
           Multi-Label Learning." Pacific-Asia Conference on Knowledge Discovery
           and Data Mining. Springer, Cham, 2018.
    """
    scoring_fn=pairwise_f1_score
    model = BinaryRelevance(LogisticRegression(solver='liblinear',  max_iter=1000,
                                                  multi_class="ovr"))
    base_model = LogisticRegression(solver='liblinear',  max_iter=1000, multi_class="ovr")
    n_models=100
    n_jobs=1
    csrpe = CSRPE(scoring_fn=scoring_fn, base_clf=base_model,
                            n_clfs=n_models, n_jobs=n_jobs, random_state=conf.al_info.exp_seed)

    labeled_pool = conf.al_info.train_logit_save.train_logits
    labeled_pool = torch.stack(labeled_pool).cpu().numpy()
    X_pool = torch.mean(logits_E_X_Y.cpu(),0).numpy()
    Y = conf.al_info.train_logit_save.train_standard_labels
    C = np.zeros(labeled_pool.shape)
    for i in range(labeled_pool.shape[0]):
        C[i, Y[i]] = 1
    Y = C
    model.train(Dataset(labeled_pool, Y))   
    csrpe.train(Dataset(labeled_pool, Y))

    predY = model.predict(X_pool)
    Z = csrpe.predicted_code(X_pool)
    predZ = csrpe.encode(predY)

    dist = paired_distances(Z, predZ, metric=hamming) # z1 z2
    dist2 = csrpe.predict_dist(X_pool) # z1 zt

    dist = dist + dist2

    topkIdx = torch.topk(torch.tensor(dist), batch_size).indices.cpu().numpy().tolist()

    return np.array(unlabeled_indices)[topkIdx].tolist()



def MultilabelWithAuxiliaryLearner(conf, logits_E_X_Y, batch_size, unlabeled_indices):
    """Multi-label Active Learning with Auxiliary Learner
    
    References
    ----------
    .. [1] Hung, Chen-Wei, and Hsuan-Tien Lin. "Multi-label Active Learning
	   with Auxiliary Learner." ACML. 2011.
    """
    n_labels = logits_E_X_Y.shape[-1]
    labeled_pool = conf.al_info.train_logit_save.train_logits
    labeled_pool = torch.stack(labeled_pool).cpu().numpy()
    X_pool = torch.mean(logits_E_X_Y.cpu(),0).numpy()
    Y = conf.al_info.train_logit_save.train_standard_labels
    C = np.zeros(labeled_pool.shape)
    for i in range(labeled_pool.shape[0]):
        C[i, Y[i]] = 1
    Y = C
    major_clf = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=conf.al_info.exp_seed)
    major_clf = BinaryRelevance(major_clf)
    major_clf.train(Dataset(labeled_pool, Y))
    aux_clf = SklearnProbaAdapter(SVC(kernel='linear',
                                probability=True,
                                gamma="auto",
                                random_state=conf.al_info.exp_seed))
    aux_clf = BinaryRelevance(aux_clf)
    aux_clf.train(Dataset(labeled_pool, Y))


    if conf.al_info.shlr.criterion == 'hlr':
        major_pred = major_clf.predict(X_pool)
        aux_pred = aux_clf.predict(X_pool)
        score = np.abs(major_pred - aux_pred).mean(axis=1)
    elif conf.al_info.shlr.criterion in ['mmr', 'shlr']:
        major_pred = major_clf.predict(X_pool) * 2 - 1

        if 'predict_real' in dir(aux_clf):
            aux_pred = aux_clf.predict_real(X_pool)
        elif 'predict_proba' in dir(aux_clf):
            aux_pred = aux_clf.predict_proba(X_pool) * 2 - 1
        else:
            raise AttributeError("aux_learner did not support either"
                                    "'predict_real' or 'predict_proba'"
                                    "method")

        # loss = (major_pred * aux_pred).mean(axis=1)
        if conf.al_info.shlr.criterion == 'mmr':
            score = (1. - major_pred * aux_pred) / 2.
            score = np.sum(score, axis=1)
        elif conf.al_info.shlr.criterion == 'shlr':
            b = conf.al_info.shlr.b
            score = (b - np.clip(major_pred * aux_pred, -b, b)) / 2. / b
            score = np.sum(score, axis=1)
        else:
            raise TypeError(
                "supported criterion are ['hlr', 'shlr', 'mmr'], the given "
                "one is: " + conf.criterion
            )

    topkIdx = torch.topk(torch.tensor(score), batch_size).indices.cpu().numpy().tolist()

    return np.array(unlabeled_indices)[topkIdx].tolist()



def AdaptiveActiveLearning(conf, logits_E_X_Y, batch_size, unlabeled_indices):
    """Adaptive Active Learning
    This approach combines Max Margin Uncertainty Sampling and Label
    Cardinality Inconsistency.
   
    References
    ----------
    .. [1] Li, Xin, and Yuhong Guo. "Active Learning with Multi-Label SVM
           Classification." IJCAI. 2013.
    """
    def _calc_approx_err(br, dataset, X_pool):
        br.train(dataset)
        br_real = br.predict_real(X_pool)

        pos = np.copy(br_real)
        pos[br_real<0] = 1
        pos = np.max((1.-pos), axis=1)

        neg = np.copy(br_real)
        neg[br_real>0] = -1
        neg = np.max((1.+neg), axis=1)

        err = neg + pos
        return np.sum(err)

    br_base = SklearnProbaAdapter(SVC(kernel='linear',
                                probability=True,
                                gamma="auto",
                                random_state=conf.al_info.exp_seed))

    # n_labels = logits_E_X_Y.shape[-1]
    X = conf.al_info.train_logit_save.train_logits
    X = torch.stack(X).cpu().numpy()
    real = torch.mean(logits_E_X_Y.cpu(),0).numpy()
    pred = torch.sigmoid(torch.tensor(real)).cpu().numpy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Y = conf.al_info.train_logit_save.train_standard_labels
    C = np.zeros(X.shape)
    for i in range(X.shape[0]):
        C[i, Y[i]] = 1
    Y = C

    # Separation Margin
    pos = np.copy(real)
    pos[real<=0] = np.inf
    neg = np.copy(real)
    neg[real>=0] = -np.inf
    separation_margin = pos.min(axis=1) - neg.max(axis=1)
    uncertainty = 1. / separation_margin

    # Label Cardinality Inconsistency
    average_pos_lbl = Y.mean(axis=0).sum()
    label_cardinality = np.sqrt((pred.sum(axis=1) - average_pos_lbl)**2)
    # pdb.set_trace()
    candidate_idx_set = set()
    betas = [i/10. for i in range(0, 11)]

    for b in betas:
        # score shape = (len(X_pool), )
        score = uncertainty**b * label_cardinality**(1.-b)
        for idx in torch.topk(torch.tensor(score), batch_size).indices.cpu().numpy():
            candidate_idx_set.add(idx)
        # for idx in np.where(score == np.max(score))[0]:
        #     candidate_idx_set.add(idx)
 
    candidates = list(candidate_idx_set)

    X_pool = torch.mean(logits_E_X_Y.cpu(),0).numpy()

    approx_err = Parallel(n_jobs=8, backend='threading')(
        delayed(_calc_approx_err)(
            BinaryRelevance(br_base),
            Dataset(np.vstack((X, X_pool[idx])), np.vstack((Y, pred[idx]))),
            X_pool)
        for idx in candidates)

    # pdb.set_trace()
    topkIdx = torch.topk(torch.tensor(approx_err), batch_size).indices.cpu().numpy().tolist()
    topkIdx = torch.tensor(candidates)[topkIdx].numpy().tolist()
    return np.array(unlabeled_indices)[topkIdx].tolist()






def MaximumLossReductionMaximalConfidence(conf, logits_E_X_Y, batch_size, unlabeled_indices):
    """Maximum loss reduction with Maximal Confidence (MMC)
    This algorithm is designed to use binary relavance with SVM as base model.
    References
    ----------
    .. [1] Yang, Bishan, et al. "Effective multi-label active learning for text
		   classification." Proceedings of the 15th ACM SIGKDD international
		   conference on Knowledge discovery and data mining. ACM, 2009.
    """
    n_labels = logits_E_X_Y.shape[-1]
    trnf = conf.al_info.train_logit_save.train_logits
    trnf = torch.stack(trnf).cpu().numpy()
    poolf = torch.mean(logits_E_X_Y.cpu(),0).numpy()
    Y = conf.al_info.train_logit_save.train_standard_labels
    C = np.zeros(trnf.shape)
    for i in range(trnf.shape[0]):
        C[i, Y[i]] = 1
    Y = C

    f = poolf * 2 - 1
    trnf = np.sort(trnf, axis=1)[:, ::-1]
    trnf /= np.tile(trnf.sum(axis=1).reshape(-1, 1), (1, trnf.shape[1]))
    if len(np.unique(Y.sum(axis=1))) == 1:
        lr = DummyClf()
    else:
        lr = LogisticRegression(random_state=conf.al_info.exp_seed)

    lr.train(Dataset(trnf, Y.sum(axis=1)))
    idx_poolf = np.argsort(poolf, axis=1)[:, ::-1]
    poolf = np.sort(poolf, axis=1)[:, ::-1]
    poolf /= np.tile(poolf.sum(axis=1).reshape(-1, 1), (1, poolf.shape[1]))
    pred_num_lbl = lr.predict(poolf).astype(int)

    yhat = -1 * np.ones((poolf.shape[0], n_labels), dtype=int)
    for i, p in enumerate(pred_num_lbl):
        yhat[i, idx_poolf[i, :p]] = 1
    score = ((1 - yhat * f) / 2).sum(axis=1)
    topkIdx = torch.topk(torch.tensor(score), batch_size).indices.cpu().numpy().tolist()
    # ask_id = seed_random_state(conf.al_info.exp_seed).choice(np.where(score == np.max(score))[0])
    return np.array(unlabeled_indices)[topkIdx].tolist()


## random a set of indices
def random_queries_batch(conf, ensemble_logitsInfo, cur_acqIdx, batch_size, unlabeled_indices):
    seed = datetime.now()
    random.seed(seed)
    fileName = conf.al_output_dir + "/seeds.txt"
    file = open(fileName, "a+")
    results = str({"acquire count": cur_acqIdx, "seeds": seed}) + "\n"
    file.writelines([results])
    file.close()

    rand_index = random.sample(range(len(ensemble_logitsInfo[0])), batch_size)
    return np.array(unlabeled_indices)[rand_index].tolist()


def reshape_3d_logits(ensemble_logitsInfo):
    logits_E_X_Y = torch.stack([torch.stack(eachE, dim=0) for eachE in ensemble_logitsInfo], dim=0)
    return logits_E_X_Y

def least_confidence_scores(conf, logits_E_X_Y, batch_size, unlabeled_indices):
    probs_X_Y = torch.nn.functional.softmax(logits_E_X_Y, dim=-1).mean(dim=0)
    lc_scores = 1 - torch.max(probs_X_Y, dim=-1).values
    topkIdx = torch.topk(lc_scores, batch_size).indices.cpu().numpy().tolist()
    return np.array(unlabeled_indices)[topkIdx].tolist()


## mean of the probability
def prob_mean(probs_X_E_Y, dim: int, keepdim: bool = False):
    return torch.mean(probs_X_E_Y, dim=dim, keepdim=keepdim)

## entropy
def entropy(probs_X_E_Y, dim: int, keepdim: bool = False):
    return -torch.sum((torch.log(probs_X_E_Y) * probs_X_E_Y).double(), dim=dim, keepdim=keepdim)


def mutual_information(probs_X_E_Y):
    sample_entropies_X_E = entropy(probs_X_E_Y, dim=-1)
    # pdb.set_trace()
    entropy_mean_X = torch.mean(sample_entropies_X_E, dim=1)

    probs_mean_X_Y = prob_mean(probs_X_E_Y, dim=1)
    mean_entropy_X = entropy(probs_mean_X_Y, dim=-1)

    mutual_info_X = mean_entropy_X - entropy_mean_X
    return mutual_info_X


def bald_scores(conf, logits_E_X_Y, batch_size, unlabeled_indices):

    probs_E_X_Y = torch.nn.functional.softmax(logits_E_X_Y, dim=-1)
    probs_X_E_Y = probs_E_X_Y.transpose(0,1) 

    bald_score_X = mutual_information(probs_X_E_Y)
    topkIdx = torch.topk(bald_score_X, batch_size).indices.cpu().numpy().tolist()

    return np.array(unlabeled_indices)[topkIdx].tolist()


## max entropy
def max_entropy_acquisition_function(conf, logits_E_X_Y, batch_size, unlabeled_indices):

    probs_E_X_Y = torch.nn.functional.softmax(logits_E_X_Y, dim=-1)
    probs_X_E_Y = probs_E_X_Y.transpose(0,1) 
    entropies_X = entropy(prob_mean(probs_X_E_Y, dim=1, keepdim=False), dim=-1)
    topkIdx = torch.topk(entropies_X, batch_size).indices.cpu().numpy().tolist()

    return np.array(unlabeled_indices)[topkIdx].tolist()


## Random generator for X prime
def random_generator_for_x_prime(probs_X_E_Y, size):
    sample_indices = sorted(random.sample(range(0, probs_X_E_Y.shape[0]), size))
    probs_Xs_E_Y = probs_X_E_Y[sample_indices,]
    return probs_Xs_E_Y


def closest_center_dist(X, centers):
    # return distance to the closest center
    dist = torch.cdist(X, X[centers])
    cd = dist.min(axis=1).values
    return cd


def kmeans_pp(X, k, centers, **kwargs):
    # kmeans++ algorithm
    if len(centers) == 0:
        # randomly choose first center
        c1 = np.random.choice(X.size(0))
        centers.append(c1)
        k -= 1

    # greedily choose centers
    for i in range(k):
        dist = closest_center_dist(X, centers) ** 2
        prob = (dist / dist.sum()).cpu().detach().numpy()
        ci = np.random.choice(X.size(0), p=prob)
        centers.append(ci)

    total_centers = len(centers)
    total_unique_centers = len(set(centers))
    m = total_centers - total_unique_centers

    if m > 0:
        pool = np.delete(np.arange(X.size(0)), list(set(centers)))
        p = random.sample(range(len(pool)), m)
        centers = np.concatenate((list(set(centers)), pool[p]), axis = None)

    return centers



## kmeans
def kmeans(rr, k):

    kmeans = KMeans(n_clusters=k, random_state=100, n_init='auto').fit(rr)
    centers = kmeans.cluster_centers_
    # find the nearest point to centers
    allMins = cdist(centers, rr)
    centroids = allMins.argmin(axis=1)
    centroids_set = np.unique(centroids)

    m = len(centroids_set)

    while k - m > 0:

        mask_centroids = []
        duplicates = []
        for x in centroids:
            if x in centroids_set and x not in duplicates:
                mask_centroids.append(True)
                duplicates.append(x)
            elif x in centroids_set and x in duplicates:
                mask_centroids.append(False)
            elif x not in centroids_set:
                mask_centroids.append(True)
        # pdb.set_trace()      
        allMins = allMins[~np.array(mask_centroids)]
        allMins[:, centroids_set] = 1000000
        centroids = allMins.argmin(axis=1)
        prev_cs = centroids_set
        centroids_set = np.unique(centroids)
     
        centroids_set = np.concatenate([prev_cs, centroids_set])
        m = len(centroids_set)
   
    return centroids_set


## cluster methods
def clustering(cl_method, rr_X_Xp, probs_X_E_Y, topT, batch_size):
    
    rr_X = torch.sum(rr_X_Xp, dim=-1)
    rr_topk_X = torch.topk(rr_X, round(probs_X_E_Y.shape[0] * topT))
    rr_topk_X_indices = rr_topk_X.indices.cpu().detach().numpy()
    rr_X_Xp = rr_X_Xp[rr_topk_X_indices]
    rr_X_Xp = normalize(rr_X_Xp)
    # rr_X_Xp = convert_embedding_by_tsne(rr_X_Xp)

    if cl_method == "kmean":
        rr = kmeans(rr_X_Xp, batch_size)
    elif cl_method == "kmeanpp":
        rr = kmeans_pp(rr_X_Xp, batch_size, [])
    rr = [rr_topk_X_indices[x] for x in rr]

    return rr


def besra(conf, logits_E_X_Y, batch_size, unlabeled_indices):

    x_prime=conf.al_info.elr_info.xprime
    split=conf.al_info.elr_info.xpsplit
    topT=conf.al_info.elr_info.topT
    alpha=float(conf.al_info.elr_info.alpha)
    beta =float(conf.al_info.elr_info.beta)

    probs_E_X_Y = torch.sigmoid(logits_E_X_Y)
    probs_E_X_Y1_1 = probs_E_X_Y.unsqueeze(-1)
    probs_E_X_Y2_1 = 1 - probs_E_X_Y1_1
    probs_E_X_Y_2 = torch.cat([probs_E_X_Y1_1, probs_E_X_Y2_1], axis=-1)
    probs_X_E_Y_2 = probs_E_X_Y_2.transpose(0,1)

    ## Generate random number of x' for all unlabel samples
    pr_YhThetaXp_Xp_E_Yh_2 = random_generator_for_x_prime(probs_X_E_Y_2, x_prime)

    ## get the maxmium value for the tag, then softmax
    ## another choice is to develop another model to classify O and non O tags

    with torch.no_grad():
     
        probs_X_E_Y_2 = probs_X_E_Y_2.cpu().detach()
        pr_YhThetaXp_Xp_E_Yh_2 = pr_YhThetaXp_Xp_E_Yh_2.cpu().detach()

        probs_X_E_Y_2_li = torch.split(probs_X_E_Y_2, split, dim=0)

        temp_rr_li = []

        for each_probs_X_E_Y_2 in probs_X_E_Y_2_li:
            ## Pr(y|theta,x)
            pr_YThetaX_X_E_Y_2 = each_probs_X_E_Y_2
            pr_ThetaL = 1 / pr_YThetaX_X_E_Y_2.shape[1]

            # Transpose dimension of Pr(y|theta,x), and calculate pr(theta|L,(x,y))
            pr_YThetaX_X_E_Y_2 = pr_ThetaL * pr_YThetaX_X_E_Y_2
            pr_YThetaX_X_Y_E_2 = torch.transpose(pr_YThetaX_X_E_Y_2, 1, 2)  ## transpose by dimension E and Y
            pr_YThetaX_X_Y_2y_E = torch.transpose(pr_YThetaX_X_Y_E_2, 2, 3)

            sum_pr_YThetaX_X_Y_2_1 = torch.sum(pr_YThetaX_X_Y_2y_E, dim=-1).unsqueeze(dim=-1)
            sum_pr_YThetaX_X_Y_2_1[torch.where(sum_pr_YThetaX_X_Y_2_1==0)] = 1e-15 ## avoid division for zero issues
            pr_ThetaLXY_X_Y_2_E = pr_YThetaX_X_Y_2y_E / sum_pr_YThetaX_X_Y_2_1

            ## Calculate pr(y_hat)
            pr_ThetaLXY_X_1_Y_2_E = pr_ThetaLXY_X_Y_2_E.unsqueeze(dim=1)
            pr_YhThetaXp_Xp_Yh_E_2 = torch.transpose(pr_YhThetaXp_Xp_E_Yh_2, 1, 2) 
            pr_Yhat_X_Xp_Y_2y_2yh = torch.matmul(pr_ThetaLXY_X_1_Y_2_E, pr_YhThetaXp_Xp_Yh_E_2) 

            ## Calculate core beta by using unsqueeze into same dimension for pr(y_hat) and pr(y_hat|theta,x)
            pr_YhThetaXp_1_1_Xp_Y_E_2yh = pr_YhThetaXp_Xp_Yh_E_2.unsqueeze(dim=0).unsqueeze(dim=0)
            pr_YhThetaXp_X_2y_Xp_Y_E_2yh = pr_YhThetaXp_1_1_Xp_Y_E_2yh.repeat(pr_Yhat_X_Xp_Y_2y_2yh.shape[0], pr_Yhat_X_Xp_Y_2y_2yh.shape[3], 1, 1, 1, 1)

            pr_Yhat_1_X_Xp_Y_2y_2yh = pr_Yhat_X_Xp_Y_2y_2yh.unsqueeze(dim=0)
            pr_Yhat_E_X_Xp_Y_2y_2yh = pr_Yhat_1_X_Xp_Y_2y_2yh.repeat(pr_YhThetaXp_Xp_Yh_E_2.shape[2], 1, 1, 1, 1, 1)
            pr_Yhat_X_2y_Xp_Y_E_2yh = pr_Yhat_E_X_Xp_Y_2y_2yh.transpose(0, 4).transpose(0, 1)

            if beta == 1.0:
                corebeta_y0 = torch.mul(pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,0] * 1 / (1 + alpha), pr_Yhat_X_2y_Xp_Y_E_2yh[:,:,:,:,:,0].pow(1 + alpha)
                            - pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,0].pow(1 + alpha))

                corebeta_y1 = torch.mul(pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1], 1 / (1 + alpha) *
                            (pr_Yhat_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1].pow(1 + alpha) - pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1].pow(1 + alpha))
                            - 1 / alpha * (pr_Yhat_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1].pow(alpha) - pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1].pow(alpha)))
            elif beta == 0 and alpha == 0:
                corebeta_y0 = torch.mul(pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,0], 
                                        torch.log((1 - pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,0])/(1- pr_Yhat_X_2y_Xp_Y_E_2yh[:,:,:,:,:,0])))
                corebeta_y1 = torch.mul(pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1], 
                                        torch.log((pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1])/(pr_Yhat_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1])))
            elif beta < 1.0:
                corebeta_y0 = torch.mul(pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,0], 1 / (beta * (1 + beta)) * ((1 - pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,0]).pow(beta)
                                        * (pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,0] * beta + 1) - (1 - pr_Yhat_X_2y_Xp_Y_E_2yh[:,:,:,:,:,0]).pow(beta) * 
                                                                                                   (pr_Yhat_X_2y_Xp_Y_E_2yh[:,:,:,:,:,0] * beta + 1)))
                corebeta_y1 = torch.mul(pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1], 1 / (1 + beta) * ((1 - pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1]).pow(beta)
                                        * (pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1] + 1) - (1 - pr_Yhat_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1]).pow(beta) * 
                                                                                                   (pr_Yhat_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1] + 1)))
            elif beta > 1.0:
                s_score_y0 = 0
                s_score_y1 = 0
                k = int(beta) - 1
                for i in range(k):
                    s_score_y0 = s_score_y0 + comb(k, i) * (-1) ** (i+1) * 1 / (alpha + i + 1) * (pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,0].pow(alpha + i + 1) - pr_Yhat_X_2y_Xp_Y_E_2yh[:,:,:,:,:,0].pow(alpha + i + 1))
            
                corebeta_y0 = torch.mul(pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,0], s_score_y0)

                k = int(beta)
                for i in range(k):
                    s_score_y1 = s_score_y1 + comb(k, i) * (-1) ** (i+1) * 1 / (alpha + i) * (pr_Yhat_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1].pow(alpha + i) - pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1].pow(alpha + i))
                
                corebeta_y1 = torch.mul(pr_YhThetaXp_X_2y_Xp_Y_E_2yh[:,:,:,:,:,1], s_score_y1)
  
              
            corebeta_X_2y_Xp_Y_E = corebeta_y0 + corebeta_y1

            corebeta_X_2y_Xp_Y = torch.sum(corebeta_X_2y_Xp_Y_E, dim=-1)

            corebeta_X_Xp_2y_Y = torch.transpose(corebeta_X_2y_Xp_Y, 1, 2)
            corebeta_Xp_X_2y_Y = torch.transpose(corebeta_X_Xp_2y_Y, 0, 1)

            pr_YLX_X_2y_Y = torch.sum(pr_YThetaX_X_Y_2y_E, dim=-1).transpose(1,2)

            rr_Xp_X_2y_Y = pr_YLX_X_2y_Y.unsqueeze(0) * corebeta_Xp_X_2y_Y
            rr_Xp_X = torch.sum(torch.sum(rr_Xp_X_2y_Y, dim=-1), dim=-1)
            rr_X_Xp = torch.transpose(rr_Xp_X, 0, 1)

            if torch.any(torch.isnan(rr_X_Xp)):
                rr_X_Xp = torch.nan_to_num(rr_X_Xp, nan=0.0, posinf=1e9, neginf=-1e9)
            temp_rr_li.append(rr_X_Xp)


        rr_X_Xp = torch.cat(temp_rr_li, dim=0)   
        if batch_size == 1:
            rr = torch.sum(rr_X_Xp, dim=-1) / pr_YhThetaXp_Xp_E_Yh_2.shape[0]
            topkIdx = torch.topk(rr, batch_size).indices.cpu().numpy().tolist()
        else:
            topkIdx = clustering(conf.al_info.cl_method, rr_X_Xp, probs_X_E_Y_2, topT, batch_size)

    return np.array(unlabeled_indices)[topkIdx].tolist()


if __name__ == '__main__':
    
    config = Config(config_file=sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.train.visible_device_list)
    config.al_info.exp_seed = int(config.al_info.exp_seed)
    config.al_info.val_info.val_seed = int(config.al_info.val_info.val_seed)
    config.al_info.val_info.val_seed = int(config.al_info.val_info.val_seed)

    logits_E_X_Y = torch.rand(5,100,159)
    batch_size = 10
    unlabeled_indices = np.arange(100)


    besra(config, logits_E_X_Y, batch_size, unlabeled_indices)