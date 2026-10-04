"""Compare actual release function, including object identity and all metadata."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from loris.discovery.patterns import _select_group_reference as original
from loris.discovery.patterns import select_group as accelerated


def items_for(seed, n=80, rows=71, labels=7, zeros=False, duplicate=False):
    rng=np.random.RandomState(seed)
    masks=rng.random_sample((n,rows)) < rng.uniform(.01,.75,(n,1))
    masks[0]=False
    if n>1:masks[1]=True
    if duplicate and n>5:masks[-5:]=masks[:5]
    corr=np.zeros((n,labels)) if zeros else rng.choice([-1.,-.5,-.1,-0.,0.,.1,.5,1.],size=(n,labels))
    result=[dict(spec=SimpleNamespace(key='pattern_%06d'%i),mask=masks[i],corr=corr[i],
                 support=int(masks[i].sum()),tp=np.arange(labels),extra={'untouched':i}) for i in range(n)]
    rng.shuffle(result)
    return result,np.zeros((rows,labels),bool)


def exact_signature(value):
    chosen,rejected=value
    return ([(id(item),item['spec'].key,method,label,sign) for item,method,label,sign in chosen],
            [(id(item),item['spec'].key,representative) for item,representative in rejected])


@pytest.mark.parametrize('quota',[0,1,2,7,40,80,120])
@pytest.mark.parametrize('zero_corr',[False,True])
@pytest.mark.parametrize('duplicates',[False,True])
def test_ties_signed_zero_duplicate_and_excess_quota(quota,zero_corr,duplicates):
    items,y=items_for(81,zeros=zero_corr,duplicate=duplicates)
    assert exact_signature(accelerated(items,quota,y))==exact_signature(original(items,quota,y))


def test_randomized_complete_order_and_metadata():
    for seed in range(150):
        rng=np.random.RandomState(seed)
        n,rows,labels=int(rng.randint(2,160)),int(rng.randint(1,170)),int(rng.randint(1,15))
        items,y=items_for(seed,n,rows,labels,duplicate=seed%2==0)
        # Include continuous non-tied correlations, sparse and full saturation cases.
        if seed%3==0:
            for item in items:item['corr']=rng.uniform(-1,1,labels)
        quota=int(rng.randint(0,n*2+1))
        assert exact_signature(accelerated(items,quota,y))==exact_signature(original(items,quota,y)), seed


def test_no_candidates_and_repeated_exact_key_stable_order():
    assert accelerated([],20,np.zeros((5,3),bool))==original([],20,np.zeros((5,3),bool))
    items,y=items_for(1,n=8,rows=9,labels=2)
    for item in items:item['spec'].key='same';item['corr'][:]=0
    assert exact_signature(accelerated(items,30,y))==exact_signature(original(items,30,y))
