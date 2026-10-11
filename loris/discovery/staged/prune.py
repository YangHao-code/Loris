"""Prune A-grown original BO programs using B, which is fitted pruning data."""
import collections,json
from . import search as s

def prune(program,rules,initial,base,rows,y):
 metric=s.Evaluator(program,rows,y);ids=list(initial);state,q=metric(ids);trace=[];base=set(base)
 heads={i:r.head.label for i,r in enumerate(rules)}
 base_by=collections.defaultdict(list)
 for i in sorted(base):base_by[heads[i]].append(i)
 while True:
  options=[]
  # Entire groups can be restored, which undoes an unsuccessful replacement.
  for head in sorted(base_by):
   candidate=[i for i in ids if heads[i]!=head]+base_by[head]
   if set(candidate)!=set(ids):options.append((dict(kind='restore_base_head',head=head),candidate))
  # Individual additions can be removed without a fixed cap by label or family.
  for i in ids:
   if i not in base:options.append((dict(kind='remove_added_clause',rule_index=i,head=heads[i]),[j for j in ids if j!=i]))
  best=None
  for oi,(op,candidate) in enumerate(options):
   _,nq=metric(candidate)
   if nq['macro_f1']<q['macro_f1']-1e-12 or nq['micro_f1']<q['micro_f1']-1e-12:continue
   gain=s.score(nq)-s.score(q)
   # Equal predictions permit deleting redundant extra rules, but not cycling
   # between equivalent head groups with the same number of changed clauses.
   oldcost=len(set(ids)-base)+len(base-set(ids));newcost=len(set(candidate)-base)+len(base-set(candidate))
   if gain<=1e-12 and newcost>=oldcost:continue
   key=(gain,oldcost-newcost,-oi)
   if best is None or key>best[0]:best=(key,op,candidate,nq)
  if best is None:break
  _,op,ids,q=best;trace.append(dict(operation=op,quality_B=q))
 return ids,trace
