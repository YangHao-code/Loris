from itertools import combinations
import glob
import numpy as np
from copy import deepcopy
from math import floor
from itertools import product
from collections import deque,OrderedDict
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
import os
import argparse
import nltk
from nltk.corpus import stopwords
nltk.download('words')

from nltk.corpus import words as nltk_words

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import copy
import pickle
import argparse
from itertools import chain
from rulecleaner_src.TreeRules import (
	textblob_sentiment,
	Node,
	Predicate,
	KeywordPredicate,
	DCAttrPredicate,
	DCConstPredicate,
	SentimentPredicate,
	PredicateNode,
	LabelNode,
	TreeRule,
	SPAM,
	HAM,
	ABSTAIN,
	CLEAN,
	DIRTY,
	textblob_sentiment
)

import pdb

nltk.download('stopwords') 
stop_words = set(stopwords.words('english')) 
nltk_words_set =  set(nltk_words.words())

logger = logging.getLogger(__name__)


def populate_violations(tree_rule, complaint):
	# given a tree rule and a complaint, populate the complaint
	# to the leaf nodes

	leaf_node = tree_rule.evaluate(complaint,'node')
	# TODO: Tentative fix to add abstain if absent
	if(ABSTAIN not in leaf_node.pairs):
		leaf_node.pairs[ABSTAIN] = []
	leaf_node.pairs[complaint['expected_label']].append(complaint)
	# print(leaf_nodes)
	# print(tree_rule)
	# print('\n')
	return leaf_node

def redistribute_after_fix(tree_rule, node, the_fix):
	# there are some possible "side effects" after repair for a pair of violations
	# which is solving one pair can simutaneously fix some other pairs so we need 
	# to redistribute the pairs in newly added nodes if possible
	sign=None
	cur_number=tree_rule.max_node_id+1

	# the_fix_words, llabel, rlabel = the_fix

	# if(reverse):
	# 	llabel, rlabel = rlabel, llabel
	possible_labels = tree_rule.possible_labels
	print(f"pssible_labels: {possible_labels}")
	print(f'tree_rule: {tree_rule}')
	

	new_predicate_node = PredicateNode(number=cur_number,pred=KeywordPredicate(keywords=[the_fix]))
	new_predicate_node.is_added=True
	cur_number+=1
	new_predicate_node.left= LabelNode(number=cur_number, pairs={k:[] for k in possible_labels}, used_predicates=set([the_fix]))
	new_predicate_node.left.is_added=True
	cur_number+=1
	new_predicate_node.right=LabelNode(number=cur_number, pairs={k:[] for k in possible_labels}, used_predicates=set([the_fix]))
	new_predicate_node.right.is_added=True
	new_predicate_node.left.parent= new_predicate_node
	new_predicate_node.right.parent= new_predicate_node
	tree_rule.max_node_id=max(cur_number, tree_rule.max_node_id)

	# reassign parent and child with new predicate node
	if(node.parent.left is node):
		node.parent.left=new_predicate_node
	else:
		node.parent.right=new_predicate_node

	new_predicate_node.parent = node.parent

	for k in possible_labels:
		for p in node.pairs[k]:
			if(new_predicate_node.pred.evaluate(p)):
				new_predicate_node.right.pairs[p['expected_label']].append(p)
				new_predicate_node.right.used_predicates.add(the_fix)
			else:
				new_predicate_node.left.pairs[p['expected_label']].append(p)
				new_predicate_node.left.used_predicates.add(the_fix)
	
	# find the dominate node 
	dominate_class_left= max(new_predicate_node.left.pairs, key=lambda k: len(new_predicate_node.left.pairs[k]))
	dominate_class_right= max(new_predicate_node.right.pairs, key=lambda k: len(new_predicate_node.right.pairs[k]))

	new_predicate_node.left.label= dominate_class_left
	new_predicate_node.right.label = dominate_class_right

	new_predicate_node.pairs={k:[] for k in possible_labels}

	return new_predicate_node


def find_available_repair(ins1, ins2, used_predicates, all_possible=False):
	"""
	given a leafnode, we want to find an attribute that can be used
	to give a pair of tuples the desired label.

	there are 2 possible resources:
	1: attribute equals/not equal attribute (NOTE: limited to the same attribute for now)
	2: attribute equals/not equals constant: 
		1. equals is just constant itself
		2. not equals is find any other value exist in the domain
	"""
	# loop through every attribute to find one
	res = []

	# find the difference between ham_pair and spam_pair
	ins1_text, ins2_text = ins1.text, ins2.text

	ins1_available_words=list(OrderedDict.fromkeys([h for h in ins1_text.split() if h not in ins2_text.split()]))
	ins2_available_words=list(OrderedDict.fromkeys([h for h in ins2_text.split() if h not in ins1_text.split()]))

	# print(f'ham_available_words: {ham_available_words}')
	# print(f'spam_available_words: {spam_available_words}')

	# exit()


	# start with attribute level and then constants
	# cand=None
	for w in ins1_available_words:
		# tuple cand has x elements: 
		# w: the word to differentate the 2 sentences
		# llabel: the left node label
		# rlabel: the right node label 
		# check if the predicate is already present
		# in the current constraint
		if(w.lower() not in stop_words and w.lower() in nltk_words_set and len(w)>1):
			if(w not in used_predicates):
				if(not all_possible):
					return w
				else:
					res.append(w)

	for w in ins2_available_words:
		# tuple cand has x elements: 
		# w: the word to differentate the 2 sentences
		# llabel: the left node label
		# rlabel: the right node label 
		# check if the predicate is already present
		# in the current constraint
		if(w.lower() not in stop_words and w.lower() in nltk_words_set and len(w)>1):
			if(w not in used_predicates):
				if(not all_possible):
					return w
				else:
					res.append(w)
	return res

def locate_node(tree, number):
	# print(tree)
	# print(f"locating node {number}")
	queue = deque([tree.root])
	while(queue):
		cur_node = queue.popleft()
		if(cur_node.number==number):
			return cur_node
		if(cur_node.left):
			queue.append(cur_node.left)
		if(cur_node.right):
			queue.append(cur_node.right)
	print('cant find the node!')
	exit()

def check_tree_purity(tree_rule, start_number=1):
	# print("checking purity...")
	# print("**********************************")
	# print(tree_rule)
	# exit()
	root = locate_node(tree_rule, start_number)
	queue = deque([root])
	leaf_nodes = []
	while(queue):
		# print(queue)
		cur_node = queue.popleft()
		if(isinstance(cur_node,LabelNode)):
			leaf_nodes.append(cur_node)
		if(cur_node.left):
			queue.append(cur_node.left)
		if(cur_node.right):
			queue.append(cur_node.right)
		# print(cur_node.number)

	for n in leaf_nodes:
		# print(f'leaf node label : {n.label}')
		for k in [HAM,SPAM,ABSTAIN]:
			for p in n.pairs[k]:
				# print(p)
				if(p['expected_label']!=n.label):
					return False
	# print("**********************************")

	return True

# def calculate_gini(node, the_fix):

# 	sign=None

# 	candidate_new_pred_node = PredicateNode(number=1,pred=KeywordPredicate(keywords=[the_fix]))
	
# 	right_leaf_spam_cnt=0
# 	right_leaf_ham_cnt=0
# 	right_leaf_abstain_cnt=0
# 	left_leaf_spam_cnt=0
# 	left_leaf_ham_cnt=0
# 	left_leaf_abstain_cnt=0

# 	for k in [SPAM, HAM, ABSTAIN]:
# 		if(k in node.pairs):
# 			for p in node.pairs[k]:
# 				if(candidate_new_pred_node.pred.evaluate(p)):
# 					if(k==ABSTAIN):
# 						right_leaf_abstain_cnt+=1
# 					if(k==SPAM):
# 						right_leaf_spam_cnt+=1
# 					else:
# 						right_leaf_ham_cnt+=1
# 				else:
# 					if(k==ABSTAIN):
# 						left_leaf_abstain_cnt+=1
# 					if(k==SPAM):
# 						left_leaf_spam_cnt+=1
# 					else:
# 						left_leaf_ham_cnt+=1


# 	left_total_cnt = left_leaf_spam_cnt+left_leaf_ham_cnt+left_leaf_abstain_cnt
# 	right_total_cnt = right_leaf_spam_cnt+right_leaf_ham_cnt+right_leaf_abstain_cnt
# 	total_cnt=left_total_cnt+right_total_cnt


# 	if(left_total_cnt!=0):
# 		L=(left_total_cnt/total_cnt)*(1-((left_leaf_spam_cnt/left_total_cnt)**2+\
# 								   (left_leaf_abstain_cnt/left_total_cnt)**2+(left_leaf_ham_cnt/left_total_cnt)**2))
# 	else:
# 		L=0
# 	if(right_total_cnt!=0):
# 		R=(right_total_cnt/total_cnt)*(1-((right_leaf_spam_cnt/right_total_cnt)**2+\
# 									(right_leaf_abstain_cnt/right_total_cnt)**2+(right_leaf_ham_cnt/right_total_cnt)**2))
# 	else:
# 		R=0
# 	gini_impurity = L+R

# 	return gini_impurity

def calculate_gini(node, the_fix):
	
	classes = list(node.pairs)
	candidate_new_pred_node = PredicateNode(number=1, pred=KeywordPredicate(keywords=[the_fix]))

	right_leaf_counts = {c: 0 for c in classes}
	left_leaf_counts = {c: 0 for c in classes}

	for k in classes:
			for p in node.pairs[k]:
				if candidate_new_pred_node.pred.evaluate(p):
					right_leaf_counts[k] += 1
				else:
					left_leaf_counts[k] += 1

	left_total_cnt = sum(left_leaf_counts.values())
	right_total_cnt = sum(right_leaf_counts.values())
	total_cnt = left_total_cnt + right_total_cnt

	def compute_gini(counts, total):
		if total == 0:
			return 0
		return (total / total_cnt) * (1 - sum((counts[c] / total) ** 2 for c in classes))

	L = compute_gini(left_leaf_counts, left_total_cnt)
	R = compute_gini(right_leaf_counts, right_total_cnt)

	gini_impurity = L + R

	return gini_impurity


def fix_violations(treerule, repair_strategy, leaf_nodes):
	# [LORIS-ADAPTER GUARD] The authors' repair splits leaves until purity, which
	# is unreachable on noisy multi-label text (their setting is binary spam/ham)
	# and grows a degenerate >1000-deep tree (RecursionError / hang). Bound the
	# tree size so the repair terminates; well-behaved rules are unaffected.
	# Override via env LORIS_RC_MAX_TREE_SIZE. Disclosed in the adapter docstring.
	import os as _os
	_MAX_TREE_SIZE = int(_os.environ.get("LORIS_RC_MAX_TREE_SIZE", "61"))
	logger.debug(f"fixing rule {treerule.__str__()}")
	if(repair_strategy=='naive'):
		# initialize the queue to work with
		queue = deque([])
		for ln in leaf_nodes:
			queue.append(ln)
		# print(queue)
		# print(len(queue))

		while(queue):
			if(treerule.size >= _MAX_TREE_SIZE):
				break
			# print(f'queue size={len(queue)}')
			node = queue.popleft()
			# print(f'node.pairs: HAM={len(node.pairs[HAM])}, SPAM={len(node.pairs[SPAM])},')
			new_parent_node=None
			node_lists = [node.pairs[SPAM], node.pairs[HAM], node.pairs[ABSTAIN]]
			# if(node.pairs[SPAM] and node.pairs[HAM]):
			if(sum(bool(lst) for lst in node_lists))>= 2:
				found=False
				the_fix=None
				for i in range(len(node_lists)):
					if(found):
						break
					for j in range(i + 1, len(node_lists)):
						for pair in list(product(node_lists[i], node_lists[j])):
							the_fix = find_available_repair(pair[0],
							pair[1], node.used_predicates)
							if(the_fix):
								found=True
								break
				new_parent_node=redistribute_after_fix(treerule, node, the_fix)


			# handle the left and right child after redistribution
			else:
				if(node.pairs[SPAM]):
					if(node.label!=SPAM):
						node.label=SPAM
						node.is_reversed=True
						treerule.reversed_cnt+=1
						treerule.setsize(treerule.size+2)
				elif(node.pairs[HAM]):
					if(node.label!=HAM):
						node.label=HAM
						node.is_reversed=True
						treerule.reversed_cnt+=1
						treerule.setsize(treerule.size+2)
				else:
					if(node.label!=ABSTAIN):
						node.label=ABSTAIN
						node.is_reversed=True
						treerule.reversed_cnt+=1
						treerule.setsize(treerule.size+2)
				# if(check_tree_purity(treerule)):
				# 	return treerule

			if(new_parent_node):
				still_inpure=False
				for k in [SPAM,HAM,ABSTAIN]:
					if(still_inpure):
						break
					for p in new_parent_node.left.pairs[k]:
						if(p['expected_label']!=new_parent_node.left.label):
							queue.append(new_parent_node.left)
							still_inpure=True
							break
				still_inpure=False          
				for k in [SPAM,HAM,ABSTAIN]:
					if(still_inpure):
						break
					for p in new_parent_node.right.pairs[k]:
							if(p['expected_label']!=new_parent_node.right.label):
								queue.append(new_parent_node.right)
								still_inpure=True
								break
				# print(queue)
				treerule.setsize(treerule.size+2)
		return treerule

	elif(repair_strategy=='information_gain'):
		# new implementation
		# 1. ignore the label of nodes at first
		# calculate the gini index of the split
		# and choose the best one as the solution
		# then based on the resulted majority expected labels
		# to assign the label for the children, if left get dirty
		# we flip the condition to preserve the ideal dc structure'
		# 2. during step 1, keep track of the used predicates to avoid
		# redundant repetitions
		queue = deque([])
		for ln in leaf_nodes:
			queue.append(ln)
		# print(queue)
		while(queue):
			if(treerule.size >= _MAX_TREE_SIZE):
				break
			node = queue.popleft()
			new_parent_node=None
			# need to find a pair of violations that get the different label
			# in order to differentiate them
			min_gini=1
			best_fix = None
			# reverse_condition=False

			# if(node.label==ABSTAIN):
			# 	continue
			node_lists = [node.pairs[SPAM], node.pairs[HAM], node.pairs[ABSTAIN]]
			# if(node.pairs[SPAM] and node.pairs[HAM]):
			if(sum(bool(lst) for lst in node_lists))>= 2:
				the_fix=None
				considered_fixes = set()
				for i in range(-1, len(node_lists)-1):
					for j in range(i + 1, len(node_lists)-1):
						for pair in list(product(node_lists[i], node_lists[j])):
							the_fixes = find_available_repair(pair[0],
							pair[1], node.used_predicates,
							all_possible=True)
							for f in the_fixes:
								# logger.debug(f"the fix: {f}")
								if(f in considered_fixes):
									continue
								gini =calculate_gini(node, f)
								# logger.debug(f"the fix: {f}: gini score: {gini}")
								considered_fixes.add(f)
								if(gini<min_gini):
									min_gini=gini
									best_fix=f
							# reverse_condition=reverse_cond
				if(best_fix):
					# if(reverse_condition):
						# pdb.set_trace()
					new_parent_node=redistribute_after_fix(treerule, node, best_fix)
			# handle the left and right child after redistribution
			else:
				if(node.pairs[SPAM]):
					if(node.label!=SPAM):
						node.label=SPAM
						node.is_reversed=True
						treerule.reversed_cnt+=1
						treerule.setsize(treerule.size+2)
				elif(node.pairs[HAM]):
					if(node.label!=HAM):
						node.label=HAM
						node.is_reversed=True
						treerule.reversed_cnt+=1
						treerule.setsize(treerule.size+2)
				else:
					if(node.label!=ABSTAIN):
						node.label=ABSTAIN
						node.is_reversed=True
						treerule.reversed_cnt+=1
						treerule.setsize(treerule.size+2)
				# if(check_tree_purity(treerule)):

			if(new_parent_node):
				still_inpure=False
				for k in [HAM,SPAM,ABSTAIN]:
					if(still_inpure):
						break
					for p in new_parent_node.left.pairs[k]:
						if(p['expected_label']!=new_parent_node.left.label):
							queue.append(new_parent_node.left)
							still_inpure=True
							break
				still_inpure=False          
				for k in [SPAM,HAM,ABSTAIN]:
					if(still_inpure):
						break
					for p in new_parent_node.right.pairs[k]:
							if(p['expected_label']!=new_parent_node.right.label):
								queue.append(new_parent_node.right)
								still_inpure=True
								break
				treerule.setsize(treerule.size+2)

				logger.debug(f"after fix, treerule size is {treerule.size}")
				if(treerule.size>1000):
					pdb.set_trace()

		logger.debug(f"fixed rule {treerule.__str__()}")
		return treerule

	elif(repair_strategy=='optimal'):
		# 1. create a queue with tree nodes
		# 2. need to deepcopy the tree in order to enumerate all possible trees
		# logger.debug("leaf_nodes:")
		# logger.debug(leaf_nodes)
		queue = deque([])
		for ln in leaf_nodes:
			queue.append(ln.number)
		# print(f"number of leaf_nodes: {len(queue)}")
		# print("queue")
		# print(queue)'
		# pdb.set_trace()
		cur_fixed_tree = treerule
		while(queue):
			sub_root_number = queue.popleft()
			subqueue=deque([(cur_fixed_tree, sub_root_number, sub_root_number)])
			# triples are needed here, since: we need to keep track of the 
			# updated(if so) subtree root in order to check purity from that node
			# print(f"subqueue: {subqueue}")
			sub_node_pure=False
			while(subqueue and not sub_node_pure):
				logger.debug(f"len of subqueue: {len(subqueue)}")
				prev_tree, leaf_node_number, subtree_root_number = subqueue.popleft()
				node = locate_node(prev_tree, leaf_node_number)

				node_lists = [node.pairs[SPAM], node.pairs[HAM], node.pairs[ABSTAIN]]
				
				if(sum(bool(lst) for lst in node_lists))>= 2:
					for i in range(len(node_lists)):
						for j in range(i + 1, len(node_lists)):
							considered_fixes = set()
							# print(list(product(node.pairs[CLEAN], node.pairs[DIRTY])))
							for pair in list(product(node_lists[i], node_lists[j])):
								if(sub_node_pure):
									break
								fixes = find_available_repair(pair[0],pair[1], node.used_predicates,all_possible=True)
								valid_fixes = [x for x in fixes if x]
								if(not valid_fixes):
									continue
								else:
									for f in valid_fixes:
										new_parent_node=None
										if(f in considered_fixes):
											continue
										considered_fixes.add(f)
										new_tree = copy.deepcopy(prev_tree)
										node = locate_node(new_tree, node.number)
										new_parent_node = redistribute_after_fix(new_tree, node, f)
										if(leaf_node_number==sub_root_number):
											# first time replacing subtree root, 
											# the node number will change so we need 
											# to replace it
											subtree_root_number=new_parent_node.number
											# print(f"subtree_root_number is being updated to {new_parent_node.number}")
										new_tree.setsize(new_tree.size+2)
										if(check_tree_purity(new_tree, subtree_root_number)):
											# print("done with this leaf node, the fixed tree is updated to")
											cur_fixed_tree = new_tree
											# print(cur_fixed_tree)
											sub_node_pure=True
											break
										# else:
										#     print("not pure yet, need to enqueue")
										# handle the left and right child after redistribution
										still_inpure=False
										for k in [HAM,SPAM,ABSTAIN]:
											if(still_inpure):
												break
											for p in new_parent_node.left.pairs[k]:
												if(p['expected_label']!=new_parent_node.left.label):
													# print("enqueued")
													# print("current_queue: ")
													new_tree = copy.deepcopy(new_tree)
													parent_node = locate_node(new_tree, new_parent_node.number)
													subqueue.append((new_tree, parent_node.left.number, subtree_root_number))
													# print(subqueue)
													still_inpure=True
													break
										still_inpure=False          
										for k in [HAM,SPAM,ABSTAIN]:
											if(still_inpure):
												break
											for p in new_parent_node.right.pairs[k]:
												if(p['expected_label']!=new_parent_node.right.label):
													# print("enqueued")
													# print("current_queue: ")
													new_tree = copy.deepcopy(new_tree)
													parent_node = locate_node(new_tree, new_parent_node.number)
													# new_parent_node=redistribute_after_fix(new_tree, new_node, f)
													subqueue.append((new_tree, parent_node.right.number, subtree_root_number))
													# print(subqueue)
													still_inpure=True
													break
										# print('\n')
				else:
					# print("just need to reverse node condition")
					if(node.pairs[SPAM]):
						if(node.label!=SPAM):
							node.label=SPAM
							node.is_reversed=True
							treerule.reversed_cnt+=1
							prev_tree.setsize(prev_tree.size+2)
					elif(node.pairs[HAM]):
						if(node.label!=HAM):
							node.label=HAM
							node.is_reversed=True
							treerule.reversed_cnt+=1
							prev_tree.setsize(prev_tree.size+2)
					else:
						if(node.label!=ABSTAIN):
							node.label=ABSTAIN
							node.is_reversed=True
							treerule.reversed_cnt+=1
							treerule.setsize(treerule.size+2)
					# if(check_tree_purity(prev_tree,subtree_root_number)):                
					# 	# print("done with this leaf node, the fixed tree is updated to")
					# 	cur_fixed_tree = prev_tree
					# 	sub_node_pure=True
					# 	# print(cur_fixed_tree)
					# 	break
				# print(f"current queue size: {len(queue)}")
		print("fixed all, return the fixed tree")
		print(cur_fixed_tree)
		return cur_fixed_tree 
	else:
		print("not a valid repair option")
		exit()

def fix_violations_multi_class(treerule, repair_strategy, leaf_nodes):
	import os as _os
	_MAX_TREE_SIZE = int(_os.environ.get("LORIS_RC_MAX_TREE_SIZE", "61"))  # [LORIS-ADAPTER GUARD]
	logger.debug(f"fixing rule {treerule.__str__()}")
	if(repair_strategy=='naive'):
		# initialize the queue to work with
		queue = deque([])
		for ln in leaf_nodes:
			queue.append(ln)
		# print(queue)
		# print(len(queue))

		while(queue):
			# print(f'queue size={len(queue)}')
			node = queue.popleft()
			# print(f'node.pairs: HAM={len(node.pairs[HAM])}, SPAM={len(node.pairs[SPAM])},')
			new_parent_node=None
			node_lists = [node.pairs[SPAM], node.pairs[HAM], node.pairs[ABSTAIN]]
			# if(node.pairs[SPAM] and node.pairs[HAM]):
			if(sum(bool(lst) for lst in node_lists))>= 2:
				found=False
				the_fix=None
				for i in range(len(node_lists)):
					if(found):
						break
					for j in range(i + 1, len(node_lists)):
						for pair in list(product(node_lists[i], node_lists[j])):
							the_fix = find_available_repair(pair[0],
							pair[1], node.used_predicates)
							if(the_fix):
								found=True
								break
				new_parent_node=redistribute_after_fix(treerule, node, the_fix)


			# handle the left and right child after redistribution
			else:
				if(node.pairs[SPAM]):
					if(node.label!=SPAM):
						node.label=SPAM
						node.is_reversed=True
						treerule.reversed_cnt+=1
						treerule.setsize(treerule.size+2)
				elif(node.pairs[HAM]):
					if(node.label!=HAM):
						node.label=HAM
						node.is_reversed=True
						treerule.reversed_cnt+=1
						treerule.setsize(treerule.size+2)
				else:
					if(node.label!=ABSTAIN):
						node.label=ABSTAIN
						node.is_reversed=True
						treerule.reversed_cnt+=1
						treerule.setsize(treerule.size+2)
				# if(check_tree_purity(treerule)):
				# 	return treerule

			if(new_parent_node):
				still_inpure=False
				for k in [SPAM,HAM,ABSTAIN]:
					if(still_inpure):
						break
					for p in new_parent_node.left.pairs[k]:
						if(p['expected_label']!=new_parent_node.left.label):
							queue.append(new_parent_node.left)
							still_inpure=True
							break
				still_inpure=False          
				for k in [SPAM,HAM,ABSTAIN]:
					if(still_inpure):
						break
					for p in new_parent_node.right.pairs[k]:
							if(p['expected_label']!=new_parent_node.right.label):
								queue.append(new_parent_node.right)
								still_inpure=True
								break
				# print(queue)
				treerule.setsize(treerule.size+2)
		return treerule

	elif(repair_strategy=='information_gain'):
		# new implementation
		# 1. ignore the label of nodes at first
		# calculate the gini index of the split
		# and choose the best one as the solution
		# then based on the resulted majority expected labels
		# to assign the label for the children, if left get dirty
		# we flip the condition to preserve the ideal dc structure'
		# 2. during step 1, keep track of the used predicates to avoid
		# redundant repetitions
		queue = deque([])
		for ln in leaf_nodes:
			queue.append(ln)
		# print(queue)
		while(queue):
			if(treerule.size >= _MAX_TREE_SIZE):
				break
			node = queue.popleft()
			new_parent_node=None
			# need to find a pair of violations that get the different label
			# in order to differentiate them
			min_gini=1
			best_fix = None
			# reverse_condition=False

			# if(node.label==ABSTAIN):
			# 	continue

			node_lists = list(node.pairs.values())
			# logger.debug(f"node_lists: {node_lists}")
			# if(node.pairs[SPAM] and node.pairs[HAM]):
			if(sum(bool(lst) for lst in node_lists))>= 2:
				the_fix=None
				considered_fixes = set()
				for i in range(0, len(node_lists)):
					for j in range(i + 1, len(node_lists)):
						for pair in list(product(node_lists[i], node_lists[j])):
							the_fixes = find_available_repair(pair[0],
							pair[1], node.used_predicates,
							all_possible=True)
							for f in the_fixes:
								# logger.debug(f"the fix: {f}")
								if(f in considered_fixes):
									continue
								gini =calculate_gini(node, f)
								# logger.debug(f"the fix: {f}: gini score: {gini}")
								considered_fixes.add(f)
								if(gini<min_gini):
									min_gini=gini
									best_fix=f
							# reverse_condition=reverse_cond
				if(best_fix):
					# if(reverse_condition):
						# pdb.set_trace()
					new_parent_node=redistribute_after_fix(treerule, node, best_fix)
			# handle the left and right child after redistribution
			else:
				for l in list(node.pairs):
					if(node.pairs[l]):
						if(node.label!=l):
							node.label=l
							node.is_reversed=True
							treerule.reversed_cnt+=1
							treerule.setsize(treerule.size+2)
				# if(check_tree_purity(treerule)):

			if(new_parent_node):
				still_inpure=False
				for k in list(new_parent_node.pairs):
					if(still_inpure):
						break
					for p in new_parent_node.left.pairs[k]:
						if(p['expected_label']!=new_parent_node.left.label):
							queue.append(new_parent_node.left)
							still_inpure=True
							break
				still_inpure=False          
				for k in list(new_parent_node.pairs):
					if(still_inpure):
						break
					for p in new_parent_node.right.pairs[k]:
							if(p['expected_label']!=new_parent_node.right.label):
								queue.append(new_parent_node.right)
								still_inpure=True
								break
				treerule.setsize(treerule.size+2)

				# logger.debug(f"after fix, treerule size is {treerule.size}")
				if(treerule.size>1000):
					pdb.set_trace()

		# logger.debug(f"fixed rule {treerule.__str__()}")
		return treerule

	elif(repair_strategy=='optimal'):
		# 1. create a queue with tree nodes
		# 2. need to deepcopy the tree in order to enumerate all possible trees
		# logger.debug("leaf_nodes:")
		# logger.debug(leaf_nodes)
		queue = deque([])
		for ln in leaf_nodes:
			queue.append(ln.number)
		# print(f"number of leaf_nodes: {len(queue)}")
		# print("queue")
		# print(queue)'
		# pdb.set_trace()
		cur_fixed_tree = treerule
		while(queue):
			sub_root_number = queue.popleft()
			subqueue=deque([(cur_fixed_tree, sub_root_number, sub_root_number)])
			# triples are needed here, since: we need to keep track of the 
			# updated(if so) subtree root in order to check purity from that node
			# print(f"subqueue: {subqueue}")
			sub_node_pure=False
			while(subqueue and not sub_node_pure):
				logger.debug(f"len of subqueue: {len(subqueue)}")
				prev_tree, leaf_node_number, subtree_root_number = subqueue.popleft()
				node = locate_node(prev_tree, leaf_node_number)

				node_lists = [node.pairs[SPAM], node.pairs[HAM], node.pairs[ABSTAIN]]
				
				if(sum(bool(lst) for lst in node_lists))>= 2:
					for i in range(len(node_lists)):
						for j in range(i + 1, len(node_lists)):
							considered_fixes = set()
							# print(list(product(node.pairs[CLEAN], node.pairs[DIRTY])))
							for pair in list(product(node_lists[i], node_lists[j])):
								if(sub_node_pure):
									break
								fixes = find_available_repair(pair[0],pair[1], node.used_predicates,all_possible=True)
								valid_fixes = [x for x in fixes if x]
								if(not valid_fixes):
									continue
								else:
									for f in valid_fixes:
										new_parent_node=None
										if(f in considered_fixes):
											continue
										considered_fixes.add(f)
										new_tree = copy.deepcopy(prev_tree)
										node = locate_node(new_tree, node.number)
										new_parent_node = redistribute_after_fix(new_tree, node, f)
										if(leaf_node_number==sub_root_number):
											# first time replacing subtree root, 
											# the node number will change so we need 
											# to replace it
											subtree_root_number=new_parent_node.number
											# print(f"subtree_root_number is being updated to {new_parent_node.number}")
										new_tree.setsize(new_tree.size+2)
										if(check_tree_purity(new_tree, subtree_root_number)):
											# print("done with this leaf node, the fixed tree is updated to")
											cur_fixed_tree = new_tree
											# print(cur_fixed_tree)
											sub_node_pure=True
											break
										# else:
										#     print("not pure yet, need to enqueue")
										# handle the left and right child after redistribution
										still_inpure=False
										for k in [HAM,SPAM,ABSTAIN]:
											if(still_inpure):
												break
											for p in new_parent_node.left.pairs[k]:
												if(p['expected_label']!=new_parent_node.left.label):
													# print("enqueued")
													# print("current_queue: ")
													new_tree = copy.deepcopy(new_tree)
													parent_node = locate_node(new_tree, new_parent_node.number)
													subqueue.append((new_tree, parent_node.left.number, subtree_root_number))
													# print(subqueue)
													still_inpure=True
													break
										still_inpure=False          
										for k in [HAM,SPAM,ABSTAIN]:
											if(still_inpure):
												break
											for p in new_parent_node.right.pairs[k]:
												if(p['expected_label']!=new_parent_node.right.label):
													# print("enqueued")
													# print("current_queue: ")
													new_tree = copy.deepcopy(new_tree)
													parent_node = locate_node(new_tree, new_parent_node.number)
													# new_parent_node=redistribute_after_fix(new_tree, new_node, f)
													subqueue.append((new_tree, parent_node.right.number, subtree_root_number))
													# print(subqueue)
													still_inpure=True
													break
										# print('\n')
				else:
					# print("just need to reverse node condition")
					if(node.pairs[SPAM]):
						if(node.label!=SPAM):
							node.label=SPAM
							node.is_reversed=True
							treerule.reversed_cnt+=1
							prev_tree.setsize(prev_tree.size+2)
					elif(node.pairs[HAM]):
						if(node.label!=HAM):
							node.label=HAM
							node.is_reversed=True
							treerule.reversed_cnt+=1
							prev_tree.setsize(prev_tree.size+2)
					else:
						if(node.label!=ABSTAIN):
							node.label=ABSTAIN
							node.is_reversed=True
							treerule.reversed_cnt+=1
							treerule.setsize(treerule.size+2)
					# if(check_tree_purity(prev_tree,subtree_root_number)):                
					# 	# print("done with this leaf node, the fixed tree is updated to")
					# 	cur_fixed_tree = prev_tree
					# 	sub_node_pure=True
					# 	# print(cur_fixed_tree)
					# 	break
				# print(f"current queue size: {len(queue)}")
		print("fixed all, return the fixed tree")
		print(cur_fixed_tree)
		return cur_fixed_tree 
	else:
		print("not a valid repair option")
		exit()

def calculate_retrained_results(complaints, new_wrongs_df, file_dir):
	the_complaints = complaints[complaints['expected_label']!=complaints['model_pred']]
	the_confirmatons = complaints[complaints['expected_label']==complaints['model_pred']]
	still_wrongs = pd.merge(the_complaints, new_wrongs_df, left_on='cid', right_on='cid', how='inner')
	# still_wrongs.to_csv(file_dir+'_still_wrongs.csv', index=False)
	not_correct_anymores = pd.merge(the_confirmatons, new_wrongs_df, left_on='cid', right_on='cid', how='inner')

	complaint_fix_rate=confirm_preserve_rate=1
	if(len(the_complaints)>0):
		complaint_fix_rate=1-len(still_wrongs)/len(the_complaints)
	if(len(the_confirmatons)>0):
		confirm_preserve_rate=1-len(not_correct_anymores)/len(the_confirmatons)

	return complaint_fix_rate, confirm_preserve_rate


def fix_rules_with_solver_input(fix_book_keeping_dict, num_class=2, repair_strategy='information_gain'):
	# user input is customized for each rule, instead of the same
	# across all the rules

	all_rules_cnt=len(fix_book_keeping_dict)
	for tid in fix_book_keeping_dict:
		treerule=fix_book_keeping_dict[tid]['rule']
		user_input = fix_book_keeping_dict[tid]['user_input']
		leaf_nodes = []
		for i, c in user_input.iterrows():
			# print("the complaint is")
			# print(c.text)
			# print(type(c.text))
			leaf_node_with_complaints = populate_violations(treerule, c)
			if(leaf_node_with_complaints not in leaf_nodes):
				# if node is already in leaf nodes, dont
				# need to add it again
				leaf_nodes.append(leaf_node_with_complaints)
		if(leaf_nodes):
			# its possible for certain rule we dont have any violations
			# if(sorted_rule_ids[current_start_id_pos]==6):
			# 	import pudb; pudb.set_trace()
			if(num_class>2):
				fixed_treerule = fix_violations_multi_class(treerule, repair_strategy, leaf_nodes)
			else:
				fixed_treerule = fix_violations(treerule, repair_strategy, leaf_nodes)

			fix_book_keeping_dict[tid]['rule']=fixed_treerule
			# print(fixed_treerule)
			fix_book_keeping_dict[tid]['after_fix_size']=fixed_treerule.size
			fixed_treerule_text = fixed_treerule.serialize()
			fix_book_keeping_dict[tid]['fixed_treerule_text']=fixed_treerule_text
		else:
			fix_book_keeping_dict[tid]['rule']=treerule
			fix_book_keeping_dict[tid]['after_fix_size']=treerule.size
			fixed_treerule_text = treerule.serialize()
			fix_book_keeping_dict[tid]['fixed_treerule_text']=fixed_treerule_text
			fixed_treerule=treerule





def calculate_retrained_results(complaints, new_wrongs_df, file_dir):
	the_complaints = complaints[complaints['expected_label']!=complaints['model_pred']]
	the_confirmatons = complaints[complaints['expected_label']==complaints['model_pred']]
	still_wrongs = pd.merge(the_complaints, new_wrongs_df, left_on='cid', right_on='cid', how='inner')
	# still_wrongs.to_csv(file_dir+'_still_wrongs.csv', index=False)
	not_correct_anymores = pd.merge(the_confirmatons, new_wrongs_df, left_on='cid', right_on='cid', how='inner')


	complaint_fix_rate=confirm_preserve_rate=1
	if(len(the_complaints)>0):
		complaint_fix_rate=1-len(still_wrongs)/len(the_complaints)
	if(len(the_confirmatons)>0):
		confirm_preserve_rate=1-len(not_correct_anymores)/len(the_confirmatons)

	return complaint_fix_rate, confirm_preserve_rate



# Function to apply stemming to a sentence
def stem_sentence(sentence, stemmer):
    words = word_tokenize(sentence)
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words


