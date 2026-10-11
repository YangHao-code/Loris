"""Native-label macro/micro F1 and multilabel completeness."""
import numpy as np

def counts(pred, truth):
    tp = (pred & truth).sum(0, dtype=np.int64)
    fp = (pred & ~truth).sum(0, dtype=np.int64)
    fn = (~pred & truth).sum(0, dtype=np.int64)
    return tp, fp, fn


def f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros_like(np.asarray(den), dtype=float), where=den != 0)


def quality(pred, truth):
    tp, fp, fn = counts(pred, truth)
    support = truth.sum(1)
    multi = support > 1
    return dict(macro_f1=float(f1(tp, fp, fn).mean()),
                micro_f1=float(f1(tp.sum(), fp.sum(), fn.sum())),
                TP=int(tp.sum()), FP=int(fp.sum()), FN=int(fn.sum()),
                Comp=float(((pred & truth).sum(1)[multi] / support[multi]).mean()) if multi.any() else 0.,
                rows=len(truth), native_labels=truth.shape[1], completeness_documents=int(multi.sum()))
