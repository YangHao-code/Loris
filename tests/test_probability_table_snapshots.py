import gc
import hashlib
import weakref
import numpy as np
import pytest
from loris.labeling import probability_tables as p
from loris._engine.typed_session import array_identity
from loris._engine.hashing import array_hash, immutable_buffer
from tests.test_labeling_runner_reuse import make, checks
from tests.test_conflict_protocol_loop import close
from loris.labeling.conflict_protocol import state_identity

@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int16,np.bool_])
@pytest.mark.parametrize('view',['C','F','stride'])
def test_exact_identity_and_owned_bytes(dtype,view):
 a=np.arange(24).reshape(4,6).astype(dtype)
 if view=='F':a=np.asfortranarray(a)
 elif view=='stride':a=a[:,::2]
 f=p.freeze_probability(a)
 assert immutable_buffer(f)
 np.testing.assert_array_equal(a,f)
 assert p.probability_identity(f)==array_identity(a)
 assert p.probability_hash(f)==array_hash(a)
 a.flat[0]=1-a.flat[0]
 assert not np.array_equal(a,f)
 with pytest.raises(ValueError):f.flags.writeable=True
 with pytest.raises(ValueError):f.flat[0]=1

@pytest.mark.parametrize('kind',['mutable','readonly_alias','readonly_memoryview'])
def test_mutable_owner_never_memoized(kind):
 a=np.arange(12,dtype=np.float32).reshape(3,4)
 if kind=='mutable':v=a
 elif kind=='readonly_alias':v=a.view();v.flags.writeable=False
 else:v=np.frombuffer(memoryview(a).toreadonly(),dtype=a.dtype).reshape(a.shape)
 one=p.probability_identity(v);hash_one=p.probability_hash(v);a[0,0]=23
 assert p.probability_identity(v)!=one and p.probability_hash(v)!=hash_one
 assert p.probability_identity(v)==array_identity(v)
 assert id(v)not in p._IDENTITIES

def test_immutable_memo_avoids_rehash_and_returns_fresh_metadata(monkeypatch):
 a=p.freeze_probability(np.ones((3,4),np.float32))
 want=p.probability_identity(a);wh=p.probability_hash(a)
 def fail(*a,**k):raise AssertionError('Content was unnecessarily rehashed')
 monkeypatch.setattr(p.hashlib,'sha256',fail)
 mutated=p.probability_identity(a);mutated['shape'][0]=999;mutated['sha256']='bad'
 assert p.probability_identity(a)==want and p.probability_hash(a)==wh

def test_metadata_changes_force_correct_rehash():
 a=p.freeze_probability(np.arange(12,dtype=np.float32));first=p.probability_hash(a)
 a.shape=(3,4)
 assert p.probability_identity(a)==array_identity(a)
 assert p.probability_hash(a)!=first
 a.dtype=np.int32
 assert p.probability_identity(a)==array_identity(a)

def test_lifetime_does_not_retain_probability_storage():
 a=p.freeze_probability(np.ones((3,4)));p.probability_identity(a);key=id(a);ref=weakref.ref(a)
 del a;gc.collect();assert ref()is None and key not in p._IDENTITIES

def test_fixed_replacement_and_mutable_alias_change_still_detected(tmp_path):
 run=make(tmp_path,'fixed')
 try:
  # Real setup has only Aux, so add a fixed core table with its original identity.
  original=p.freeze_probability(np.zeros((run.n,run.k),np.float32));run.tables['fixed']=original
  run.initial_tables['fixed']=p.probability_identity(original)
  run.tables_for(run.version)
  alias=np.zeros_like(original);view=alias.view();view.flags.writeable=False
  run.tables['fixed']=view;run.tables_for(run.version)
  alias[0,0]=.25
  with pytest.raises(RuntimeError,match='fixed core'):run.tables_for(run.version)
 finally:close(run)

def test_real_checks_unchanged_against_mutable_tables(tmp_path):
 frozen=make(tmp_path,'frozen');mutable=make(tmp_path,'mutable')
 mutable.tables={k:v.copy()for k,v in mutable.tables.items()};mutable.version.probabilities=mutable.tables[mutable.alias]
 try:
  a=checks(frozen);b=checks(mutable)
  for left,right in [(a[0],b[0])]+[(x[0],y[0])for x,y in zip(a[2],b[2])]:
   np.testing.assert_array_equal(left.positive_facts,right.positive_facts)
   np.testing.assert_array_equal(left.negative_facts,right.negative_facts)
   np.testing.assert_array_equal(left.conflict_unknown,right.conflict_unknown)
   assert state_identity(left)==state_identity(right)
   assert left.reference_audit==right.reference_audit
 finally:close(frozen);close(mutable)

@pytest.mark.parametrize('shape',[(0,4),(4,0),()])
def test_empty_and_scalar_identities_match(shape):
 a=np.zeros(shape,dtype=np.float32);f=p.freeze_probability(a)
 assert p.probability_identity(f)==array_identity(a)
 assert p.probability_hash(f)==array_hash(a)

def test_same_shape_data_rebinding_cannot_reuse_old_digest():
 a=p.freeze_probability(np.arange(4,dtype=np.float32));first=p.probability_identity(a)
 old_owner=p._IDENTITIES[id(a)][3]
 with pytest.warns(DeprecationWarning):
  a.data=memoryview(np.arange(4,8,dtype=np.float32).tobytes())
 a.flags.writeable=False
 assert immutable_buffer(a)
 assert p.probability_identity(a)==array_identity(a)
 assert p.probability_identity(a)!=first
 assert p._IDENTITIES[id(a)][3] is not old_owner
 assert p.probability_hash(a)==array_hash(a.copy())

def test_same_bytes_owner_different_slice_rebinding():
 raw=np.arange(8,dtype=np.float32).tobytes();a=np.frombuffer(raw,dtype=np.float32,count=4)
 first=p.probability_identity(a);owner=p._IDENTITIES[id(a)][3]
 with pytest.warns(DeprecationWarning):
  a.data=memoryview(raw)[16:]
 a.flags.writeable=False
 assert p._immutable_identity(a)[0]is owner
 assert p.probability_identity(a)!=first
 assert p.probability_identity(a)==array_identity(a)

def test_global_digest_cache_metadata_and_data_rebind():
 from loris._engine.rlearner import array_identity as learner_identity
 a=p.freeze_probability(np.arange(8,dtype=np.float32));first=array_hash(a);learner_identity(a)
 a.shape=(2,4)
 assert array_hash(a)==array_hash(a.copy()) and array_hash(a)!=first
 with pytest.warns(DeprecationWarning):
  a.data=memoryview(np.arange(8,16,dtype=np.float32).tobytes())
 a.flags.writeable=False
 assert array_hash(a)==array_hash(a.copy())
 assert learner_identity(a)==learner_identity(a.copy())

def test_real_runner_data_rebind_cannot_reuse_context_identity(tmp_path):
 run=make(tmp_path,'rebind')
 try:
  _,before,_=run.chase(run.version,phase='initial_zero_update')
  a=run.version.probabilities;replacement=np.full(a.shape,.97,dtype=a.dtype)
  with pytest.warns(DeprecationWarning):
   a.data=memoryview(replacement.tobytes())
  a.flags.writeable=False
  _,after,_=run.chase(run.version,phase='initial_zero_update')
  assert after.runner is not before.runner
  assert after.runner.context_id!=before.runner.context_id
 finally:close(run)
