import numpy as np
import math
import copy



def randperm(n, k=None):
    """Generate a random array which contains k elements range from (n[0]:n[1])
    Parameters
    ----------
    n: int or tuple
        range from [n[0]:n[1]], include n[0] and n[1].
        if an int is given, then n[0] = 0
    k: int, optional (default=end - start + 1)
        how many numbers will be generated. should not larger than n[1]-n[0]+1,
        default=n[1] - n[0] + 1.
    Returns
    -------
    perm: list
        the generated array.
    """
    if isinstance(n, np.generic):
        # n = np.asscalar(n)  # deprecated in numpy v1.16
        n = n.item()
    if isinstance(n, tuple):
        if n[0] is not None:
            start = n[0]
        else:
            start = 0
        end = n[1]
    elif isinstance(n, int):
        start = 0
        end = n
    else:
        raise TypeError("n must be tuple or int.")

    if k is None:
        k = end - start + 1
    if not isinstance(k, int):
        raise TypeError("k must be an int.")
    if k > end - start + 1:
        raise ValueError("k should not larger than n[1]-n[0]+1")

    randarr = np.arange(start, end + 1)
    np.random.shuffle(randarr)
    return randarr[0:k]


class _LabelRankingModel_MatlabVer:
    """Label ranking model is a classification model in multi-label setting.
    It combines label ranking with threshold learning, and use SGD to optimize.
    This class is implemented strictly according to the matlab code provided
    by the author. So it's hard to use, but it guarantees correctness.
    Parameters
    ----------
    init_X: 2D array
        Feature matrix of the initial data for training.
        Shape is n*d, one row for an instance with d features.
    init_y: 2D array
        Label matrix of the initial data for training.
        Shape is n*n_classes, one row for an instance, -1 means irrelevant,
        a positive value means relevant, the larger, the more relevant.
    References
    ----------
    [1] S.-J. Huang and Z.-H. Zhou. Active query driven by uncertainty and
        diversity for incremental multi-label learning. In Proceedings
        of the 13th IEEE International Conference on Data Mining, pages
        1079-1084, Dallas, TX, 2013.
    """

    def __init__(self, init_X=None, init_y=None):
        self._init_flag = False
        if init_X is not None and init_y is not None:
            assert len(init_X) == len(init_y)
            assert len(np.shape(init_y)) == 2
            self._init_X = np.asarray(init_X)
            self._init_y = np.asarray(init_y)

            if len(np.nonzero(self._init_y == 2.0)[0]) == 0:
                self._init_y = np.hstack((self._init_y, 2 * np.ones((self._init_y.shape[0], 1))))
                # B, V, AB, AV, Anum, trounds, costs, norm_up, step_size0, num_sub, lmbda, avg_begin, avg_size, n_repeat, \
                # max_query = self.init_model_train(self._init_X, self._init_y)
            self._init_flag = True

    def get_BV(self, AB, AV, Anum):
        bv = (AV / Anum).T.dot(AB / Anum)
        return bv

    def init_model_train(self, init_data=None, init_targets=None, n_repeat=10):
        if init_data is None:
            init_data = self._init_X
        if init_targets is None:
            init_targets = self._init_y
        init_data = np.asarray(init_data)
        init_targets = np.asarray(init_targets)
        if len(np.nonzero(init_targets == 2.0)[0]) == 0:
            init_targets = np.hstack((init_targets, 2 * np.ones((init_targets.shape[0], 1))))

        tar_sh = np.shape(init_targets)
        d = np.shape(init_data)[1]
        n_class = tar_sh[1]
        # n_repeat = 10
        max_query = math.floor(tar_sh[0] * (tar_sh[1] - 1) / 2)
        D = 200
        num_sub = 5
        norm_up = np.inf
        lmbda = 0
        step_size0 = 0.05
        avg_begin = 10
        avg_size = 5

        costs = 1.0 / np.arange(start=1, stop=n_class * 5 + 1)
        for k in np.arange(start=1, stop=n_class * 5):
            costs[k] = costs[k - 1] + costs[k]

        V = np.random.normal(0, 1 / np.sqrt(d), (D, d))
        B = np.random.normal(0, 1 / np.sqrt(d), (D, n_class * num_sub))
        # import scipy.io as scio
        # ld = scio.loadmat('F:\\alipy_doc\\alipy-additional-methods-source\\multi label\\AURO\\BV_val.mat')
        # V = ld['V']
        # B = ld['B']

        for k in range(d):
            tmp1 = V[:, k]
            if np.all(tmp1 > norm_up):
                V[:, k] = tmp1 * norm_up / np.linalg.norm(tmp1)
        for k in range(n_class * num_sub):
            tmp1 = B[:, k]
            if np.all(tmp1 > norm_up):
                B[:, k] = tmp1 * norm_up / np.linalg.norm(tmp1)

        AB = 0
        AV = 0
        Anum = 0
        trounds = 0

        for rr in range(n_repeat):
            B, V, AB, AV, Anum, trounds = self.train_model(init_data, init_targets, B, V, costs, norm_up, step_size0,
                                                           num_sub, AB, AV, Anum, trounds, lmbda, avg_begin, avg_size)

        return B, V, AB, AV, Anum, trounds, costs, norm_up, step_size0, num_sub, lmbda, avg_begin, avg_size, n_repeat, max_query

    def train_model(self, data, targets, B, V, costs, norm_up, step_size0, num_sub, AB, AV, Anum, trounds,
                    lmbda, average_begin, average_size):
        """targets: 0 unlabeled, 1 positive, -1 negative, 2 dummy, 0.5 less positive"""
        targets = np.asarray(targets)
        # print(np.nonzero(targets == 2.0))
        if len(np.nonzero(targets == 2.0)[0]) == 0:
            targets = np.hstack((targets, 2 * np.ones((targets.shape[0], 1))))
        data = np.asarray(data)
        B = np.asarray(B)
        V = np.asarray(V)

        n, n_class = np.shape(targets)
        row_ind, col_ind = np.nonzero(targets >= 1)
        train_pairs = np.hstack((row_ind.reshape((-1,1)), col_ind.reshape((-1,1))))

        # tmpnums = np.sum(targets >= 1, axis=1)
        # train_pairs = np.zeros((sum(tmpnums), 1))
        # tmpidx = 0

        # for i in range(n):
        #     train_pairs[tmpidx: tmpidx + tmpnums[i]] = i+1
        #     tmpidx = tmpidx + tmpnums[i]

        # targets = targets.T
        # # tp = np.nonzero(targets.flatten() >= 1)
        # # print(tp[0])
        # # print(len(tp[0]))
        # train_pairs = np.hstack(
        #     (train_pairs,
        #      np.reshape([nz % n_class for nz in np.nonzero(targets.flatten(order='F') >= 1)[0]], newshape=(-1, 1))))
        # # train_pairs[np.nonzero(train_pairs[:, 1] == 0)[0], 1] = n_class
        # targets = targets.T

        n = np.shape(train_pairs)[0]

        random_idx = randperm(n - 1)
        # import scipy.io as scio
        # ld = scio.loadmat('F:\\alipy_doc\\alipy-additional-methods-source\\multi label\\AURO\\perm.mat')
        # random_idx = ld['random_idx'].flatten()-1

        for i in range(n):
            idx_ins = int(train_pairs[random_idx[i], 0])
            xins = data[int(idx_ins), :].T
            idx_class = int(train_pairs[random_idx[i], 1])
            if idx_class == n_class-1:
                idx_irr = np.nonzero(targets[idx_ins, :] == -1)[0]
            # elif idx_class == idxPs[idx_ins]:
            #     idx_irr = np.hstack((np.nonzero(targets[idx_ins, :] == -1)[0], int(idxNs[idx_ins]), n_class - 1))
            else:
                idx_irr = np.hstack((np.nonzero(targets[idx_ins, :] == -1)[0], n_class - 1))
            n_irr = len(idx_irr)

            By = B[:, idx_class * num_sub: (idx_class + 1) * num_sub]
            Vins = V.dot(xins)
            fy = np.max(By.T.dot(Vins), axis=0)
            idx_max_class = np.argmax(By.T.dot(Vins), axis=0)
            By = By[:, idx_max_class]
            fyn = np.NINF
            for j in range(n_irr):
                idx_pick = idx_irr[randperm(n_irr - 1, 1)[0]]
                # print(idx_irr, idx_pick)
                Byn = B[:, idx_pick * num_sub: (idx_pick + 1) * num_sub]
                # [fyn, idx_max_pick] = max(Byn.T.dot(Vins),[],1)
                # if Byn == []:
                #     print(0)
                tmp1 = Byn.T.dot(Vins)
                fyn = np.max(tmp1, axis=0)
                idx_max_pick = np.argmax(tmp1, axis=0)

                if fyn > fy - 1:
                    break

            if fyn > fy - 1:
                step_size = step_size0 / (1 + lmbda * trounds * step_size0)
                trounds = trounds + 1
                Byn = B[:, idx_pick * num_sub + idx_max_pick]
                loss = costs[math.floor(n_irr / (j + 1)) - 1]
                tmp1 = By + step_size * loss * Vins
                tmp3 = np.linalg.norm(tmp1)
                if tmp3 > norm_up:
                    tmp1 = tmp1 * norm_up / tmp3
                tmp2 = Byn - step_size * loss * Vins
                tmp3 = np.linalg.norm(tmp2)
                if tmp3 > norm_up:
                    tmp2 = tmp2 * norm_up / tmp3
                V -= step_size * loss * (
                    B[:, [idx_pick * num_sub + idx_max_pick, idx_class * num_sub + idx_max_class]].dot(
                        np.vstack((xins, -xins))))

                norms = np.linalg.norm(V, axis=0)
                idx_down = np.nonzero(norms > norm_up)[0]
                B[:, idx_class * num_sub + idx_max_class] = tmp1
                B[:, idx_pick * num_sub + idx_max_pick] = tmp2
                if idx_down.size > 0:
                    norms = norms[norms > norm_up]
                    for k in range(len(idx_down)):
                        V[:, idx_down[k]] = V[:, idx_down[k]] * norm_up / norms[k]
            if i == 0 or (trounds > average_begin and i % average_size == 0):
                AB = AB + B
                AV = AV + V
                Anum = Anum + 1

        return B, V, AB, AV, Anum, trounds

    def lr_predict(self, BV, data, num_sub):
        BV = np.asarray(BV)
        data = np.asarray(data)

        fs = data.dot(BV)
        n = data.shape[0]
        n_class = int(fs.shape[1] / num_sub)
        pres = np.ones((n, n_class)) * np.NINF
        for j in range(num_sub):
            f = fs[:, j: fs.shape[1]: num_sub]
            assert (np.all(f.shape == pres.shape))
            pres = np.fmax(pres, f)
        labels = -np.ones((n, n_class - 1))
        for line in range(n_class - 1):
            gt = np.nonzero(pres[:, line] > pres[:, n_class - 1])[0]
            labels[gt, line] = 1
        return pres, labels

class LabelRankingModel(_LabelRankingModel_MatlabVer):
    """Label ranking model is a classification model in multi-label setting.
    It combines label ranking with threshold learning, and use SGD to optimize.
    This class re-encapsulate the _LabelRankingModel_MatlabVer class for
    better use.
    It accept 3 types of labels:
    1 : relevant
    0.5 : less relevant
    -1 : irrelevant
    The labels in algorithms mean:
    2 : dummy
    0 : unknown (not use this label when updating)
    This class is mainly used for AURO and AUDI method for multi label querying.
    !! IMPORTANT
    1. This model is scaling sensitive. If you find the optimization process is not converge,
    (e.g., ZeroDivisionError: division by zero, RuntimeWarning: overflow encountered in multiply,
    or the values of BV tend to infinite.) please try to normalize your data first.
    Parameters
    ----------
    init_X: 2D array, optional (default=None)
        Feature matrix of the initial data for training.
        Shape is n*d, one row for an instance with d features.
    init_y: 2D array, optional (default=None)
        Label matrix of the initial data for training.
        Shape is n*n_classes, one row for an instance, -1 means irrelevant,
        a positive value means relevant, the larger, the more relevant.
    References
    ----------
    [1] S.-J. Huang and Z.-H. Zhou. Active query driven by uncertainty and
        diversity for incremental multi-label learning. In Proceedings
        of the 13th IEEE International Conference on Data Mining, pages
        1079-1084, Dallas, TX, 2013.
    """

    def __init__(self, init_X=None, init_y=None, **kwargs):
        super(LabelRankingModel, self).__init__(init_X, init_y)
        self._ini_parameters = None
        if self._init_flag is True:
            n_repeat = kwargs.pop('n_repeat', 10)
            self._ini_parameters = self.init_model_train(self._init_X, self._init_y, n_repeat=n_repeat)
            self._B, self._V, self._AB, self._AV, self._Anum, self._trounds, self._costs, self._norm_up, \
            self._step_size0, self._num_sub, self._lmbda, self._avg_begin, self._avg_size, self._n_repeat, \
            self._max_query = self._ini_parameters

    def fit(self, X, y, n_repeat=10, is_incremental=False):
        """Train the model from X and y.
        Parameters
        ----------
        X: 2D array, optional (default=None)
            Feature matrix of the whole dataset.
        y: 2D array, optional (default=None)
            Label matrix of the whole dataset.
        n_repeat: int, optional (default=10)
            The number of optimization iterations.
        is_incremental: bool, optional (default=False)
            Whether to train the model in an incremental way.
        """
        if not self._init_flag or not self._ini_parameters:
            self._ini_parameters = self.init_model_train(X, y, n_repeat=n_repeat)
            self._B, self._V, self._AB, self._AV, self._Anum, self._trounds, self._costs, self._norm_up, \
            self._step_size0, self._num_sub, self._lmbda, self._avg_begin, self._avg_size, self._n_repeat, \
            self._max_query = self._ini_parameters
            self._init_flag = True

        if not is_incremental:
            self._B, self._V, self._AB, self._AV, self._Anum, self._trounds, self._costs, self._norm_up, \
            self._step_size0, self._num_sub, self._lmbda, self._avg_begin, self._avg_size, self._n_repeat, \
            self._max_query = self._ini_parameters

        for i in range(n_repeat):
            self._B, self._V, self._AB, self._AV, self._Anum, self._trounds = self.train_model(
                X, y, self._B, self._V, self._costs, self._norm_up,
                self._step_size0, self._num_sub, self._AB, self._AV, self._Anum, self._trounds,
                self._lmbda, self._avg_begin, self._avg_size)

    def predict(self, X):
        BV = self.get_BV(self._AB, self._AV, self._Anum)
        return self.lr_predict(BV, X, self._num_sub)