from rulecleaner_src.TreeRules import (
	textblob_sentiment,
	Node,
	Predicate,
	KeywordPredicate,
	StemPredicate,
	RegexPredicate,
	POSPredicate,
	DCAttrPredicate,
	DCConstPredicate,
	SentimentPredicate,
	PredicateNode,
	SLengthPredicate,
	LabelNode,
	TreeRule,
	SPAM,
	HAM,
	ABSTAIN,
	CLEAN,
	DIRTY,
)
from itertools import product
from collections import deque
from datetime import datetime
import psycopg2 
import time
from snorkel.labeling import (
	LabelingFunction, 
	labeling_function, 
	PandasLFApplier, 
	LFAnalysis,
	filter_unlabeled_dataframe
	)
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
from typing import *
import pandas as pd
import logging
import random


# keyword lfs
def keyword_labelling_func_builder(keywords: List[str], expected_label: int, is_good: bool=True, possible_labels=[-1,0,1]):
	cur_number=1
	tree_size=1
	r1 = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=keywords))
	cur_number+=1
	tree_size+=1
	r1_l = LabelNode(number=cur_number, label=ABSTAIN,  pairs={k:[] for k in possible_labels}, used_predicates=set(keywords))
	tree_size+=1
	cur_number+=1
	r1_r = LabelNode(number=cur_number, label=expected_label, pairs={k:[] for k in possible_labels}, used_predicates=set(keywords))
	r1.right=r1_r
	r1.left=r1_l

	r1_l.parent=r1
	r1_r.parent=r1

	return TreeRule(rtype='lf', root=r1, size=tree_size, max_node_id=3, is_good=is_good, possible_labels=possible_labels)

# stem lfs 
def stem_labelling_func_builder(stems: List[str], expected_label: int, is_good: bool=True):
	cur_number=1
	tree_size=1
	r1 = PredicateNode(number=cur_number, pred=StemPredicate(stems=stems))
	cur_number+=1
	tree_size+=1
	r1_l = LabelNode(number=cur_number, label=ABSTAIN,  pairs={SPAM:[], HAM:[]}, used_predicates=set(stems))
	tree_size+=1
	cur_number+=1
	r1_r = LabelNode(number=cur_number, label=expected_label, pairs={SPAM:[], HAM:[]}, used_predicates=set(stems))
	r1.right=r1_r
	r1.left=r1_l

	r1_l.parent=r1
	r1_r.parent=r1

	return TreeRule(rtype='lf', root=r1, size=tree_size, max_node_id=3, is_good=is_good)


# regex lfs
def regex_func_builder(patterns: List[str], expected_label: int):
	cur_number=1
	tree_size=1
	r1 = PredicateNode(number=cur_number, pred=RegexPredicate(patterns=patterns))
	cur_number+=1
	tree_size+=1
	r1_l = LabelNode(number=cur_number, label=ABSTAIN,  pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	tree_size+=1
	cur_number+=1
	r1_r = LabelNode(number=cur_number, label=expected_label, pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	r1.right=r1_r
	r1.left=r1_l

	r1_l.parent=r1
	r1_r.parent=r1

	return TreeRule(rtype='lf', root=r1, size=tree_size, max_node_id=3)

def senti_func_builder(thresh):
	##  subjectivity rule
	cur_number=1
	tree_size=1
	r_senti = PredicateNode(number=cur_number, pred=SentimentPredicate(thresh=thresh, sent_func=textblob_sentiment))
	cur_number+=1
	tree_size+=1
	r_senti_l = LabelNode(number=cur_number, label=ABSTAIN,  pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	r_senti_r = LabelNode(number=cur_number, label=HAM, pairs={SPAM:[], HAM:[]}, used_predicates=set([]))

	r_senti.right=r_senti_r
	r_senti.left=r_senti_l
	r_senti_l.parent=r_senti
	r_senti_r.parent=r_senti

	return TreeRule(rtype='lf', root=r_senti, size=tree_size, max_node_id=3)


def pos_func_builder(tags):
	##  subjectivity rule
	cur_number=1
	tree_size=1
	r_tag = PredicateNode(number=cur_number, pred=POSPredicate(tags=tags))
	cur_number+=1
	tree_size+=1
	r_tag_l = LabelNode(number=cur_number, label=ABSTAIN,  pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	r_tag_r = LabelNode(number=cur_number, label=SPAM,  pairs={SPAM:[], HAM:[]}, used_predicates=set([]))

	r_tag.right=r_tag_r
	r_tag.left=r_tag_l
	r_tag_l.parent=r_tag
	r_tag_r.parent=r_tag

	return TreeRule(rtype='lf', root=r_tag, size=tree_size, max_node_id=3)


def length_func_builder(thresh):
	##  sentence length
	cur_number=1
	tree_size=1
	r_length = PredicateNode(number=cur_number, pred=SLengthPredicate(thresh=thresh))
	cur_number+=1
	tree_size+=1
	r_length_l = LabelNode(number=cur_number, label=ABSTAIN,  pairs={SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	r_length_r = LabelNode(number=cur_number, label=HAM, pairs={SPAM:[], HAM:[]},  used_predicates=set([]))

	r_length.right=r_length_r
	r_length.left=r_length_l
	r_length_l.parent=r_length
	r_length_r.parent=r_length

	return TreeRule(rtype='lf', root=r_length, size=tree_size, max_node_id=3)