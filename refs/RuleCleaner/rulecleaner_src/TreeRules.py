from textblob import TextBlob
from collections import deque
from typing import *
from dataclasses import dataclass
import re 
from snorkel.labeling import LabelingFunction
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from string import Template
import logging

logger = logging.getLogger(__name__)

dot_string_template=Template("""digraph "rule" { $nodes_details } """)

SPAM=1
HAM=0
ABSTAIN=-1
CLEAN=HAM
DIRTY=SPAM

def textblob_sentiment(x) -> float:
    scores = TextBlob(x.text)
    # print(scores.sentiment.subjectivity)
    return scores.sentiment.subjectivity


@dataclass
class Node:
	"""
	node structure used in the tree rule
	"""
	number: int=0
	left: 'Node'=None
	right: 'Node'=None
	parent: 'Node'=None
	is_added: bool=False
	is_reversed: bool=False

class Predicate:
	def __init__(self, instance: str):
		pass

	def evaluate(self):
		pass

class KeywordPredicate(Predicate):

	def __init__(self, keywords:List[str]):
		self.keywords=keywords
		self.pred_identifier= f"keyword_predicate-word-({','.join(self.keywords)})"

	def __repr__(self):
		return f"keyword_predicate-word-({','.join(self.keywords)})"

	def __str__(self):
		return f"keyword_predicate-word-({','.join(self.keywords)})"

	def evaluate(self,instance: str):
		return any(x in instance.text for x in self.keywords)
		# return any(x in instance.get('text', '') for x in self.keywords)
	
class StemPredicate(Predicate):

	def __init__(self, stems:List[str]):
		self.stems=stems
		self.pred_identifier= f"keyword_predicate-word-({','.join(self.stems)})"

	def __repr__(self):
		return f"Stem_predicate-stem-({','.join(self.stems)})"

	def __str__(self):
		return f"Stem_predicate-stem-({','.join(self.stems)})"

	def evaluate(self,instance: str):
		# Tokenize the sentence into words
		instance_stems=instance.stems
		return all([s in instance_stems for s in self.stems])
	

class SLengthPredicate(Predicate):

	def __init__(self, thresh):
		self.thresh=thresh
		self.pred_identifier=f"SLengthPredicate-thresh-{self.thresh}"
		
	def __repr__(self):
		return f"SLengthPredicate-thresh-{self.thresh}"

	def __str__(self):
		return f"SLengthPredicate-thresh-{self.thresh}"

	def evaluate(self, instance: str):
		"""VB with PRP$(subscribe my channel)"""
		return True if(len(instance.text.split()) < self.thresh) else False

class RegexPredicate(Predicate):

	def __init__(self, patterns):
		self.patterns=patterns
		self.pred_identifier=f"RegexPredicate-tag-({','.join(self.patterns)})"

	def __repr__(self):
		return f"RegexPredicate-tag-({','.join(self.patterns)})"

	def __str__(self):
		return f"RegexPredicate-tag-({','.join(self.patterns)})"

	def evaluate(self, instance: str):
		"""VB with PRP$(subscribe my channel)"""
		return any(re.search(t, instance.text, flags=re.I) for t in self.patterns)

class POSPredicate(Predicate):

	def __init__(self, tags: str):
		self.tags=tags
		self.pred_identifier= f"POSPredicate-tag-({','.join(self.tags)})"

	def __repr__(self):
		return f"POSPredicate-tag-({','.join(self.tags)})"

	def __str__(self):
		return f"POSPredicate-tag-({','.join(self.tags)})"

	def evaluate(self,instance: str):
		"""VB with PRP$(subscribe my channel)"""
		text_str = nltk.word_tokenize(instance.text)
		posstr = ''.join([t[1] for t in nltk.pos_tag(text_str)])
		return any(x in posstr for x in self.tags)

class DCAttrPredicate(Predicate):
	def __init__(self, pred:str, operator:str):
		self.pred=pred
		self.operator=operator
		# print(f'self.operator: {self.operator}')
		# print(self.pred)
		# print(re.findall(r'(t[12])\.[-\w]+,(t[12])\.[-\w]+', self.pred))
		# print(self.pred)
		self.pred_identifier= re.findall(r'(t[12])\.([-\w]+),(t[12])\.([-\w]+)', self.pred)[0] + (self.operator,)
		self.attr=self.pred_identifier[1].lower()

	def __repr__(self):
		return f"dc-attr-pred-{self.pred}"
	def __str__(self):
		return f"dc-attr-pred-{self.pred}"
	def evaluate(self, dict_to_eval:dict):
		# logger.critical(f'dict_to_eval in dc attr: {dict_to_eval}')
		# logger.critical(f"dict_to_eval in dc attr: dict_to_eval['t1']['{self.attr}']{self.operator}dict_to_eval['t2']['{self.attr}']")
		return eval(f"dict_to_eval['t1']['{self.attr}']{self.operator}dict_to_eval['t2']['{self.attr}']")
	    # attr = re.search(r't[1|2]\.([-\w]+)', self.pred).group(1)
	    # print(f"dict_to_eval['t1']['{self.attr}']{self.operator}dict_to_eval['t2']['{self.attr}']")	    

class DCConstPredicate(Predicate):
	def __init__(self, pred:str, operator:str):
		self.pred=pred
		self.operator=operator
		# print(self.pred)
		self.pred_identifier = re.findall(r'(t[1|2])\.([-\w]+),[\"|\'](.*)[\"|\']',self.pred)[0]+ (self.operator,)
		self.role, self.attr, self.const, _ = self.pred_identifier
		self.attr=self.attr.lower()

	def __repr__(self):
		return f"dc-const-pred-{self.pred}"
	
	def __str__(self):
		return f"dc-const-pred-{self.pred}"
	
	def evaluate(self, dict_to_eval:dict):
		# triples where each triple is in the format of (t1, attr, constant) (or t2)
		logger.critical(f'dict_to_eval in dc const: {dict_to_eval}')
		logger.critical(f"dict_to_eval in dc const: dict_to_eval['{self.role}']['{self.attr}']{self.operator}'{self.const}'")

		return eval(f"dict_to_eval['{self.role}']['{self.attr}']{self.operator}'{self.const}'")

class SentimentPredicate(Predicate):
	def __init__(self, thresh:float=0.5, sent_func:Callable=textblob_sentiment):
		self.thresh=thresh
		self.sent_func=sent_func
		self.pred_identifier= f"sentiment_predicate-thresh-{self.thresh}"

	def __repr__(self):
		return f"sentiment_predicate-thresh-{self.thresh}"

	def __str__(self):
		return f"sentiment_predicate-thresh-{self.thresh}"

	def evaluate(self, instance: str):
		return self.sent_func(instance)>self.thresh


@dataclass
class PredicateNode(Node):
	pred: Predicate=None


@dataclass
class LabelNode(Node):
	label: int=ABSTAIN
	pairs: dict=None
	used_predicates: set([])=None		

# @dataclass
# DenialPredicate(Predicate):
# 	accessedtuples: List(int)

class TreeRule:
# 	"""
# 	tree like structure of rules
# 	some properties of the tree structure
# 	1. type:
# 		'lf': labelling function rule (snorkel flavor)
# 		'dc': denial constraints (holoclean DC flavor)
# 	2. __str__ / __repr__ string representation of the rules 
# 	"""
	rule_counter=0

	# @classmethod
	# def eval_rule(ls, rule, instance):
	# 	return rule.evaluate(instance)

	def __init__(self, rtype: str, root: 'Node', size: int, max_node_id: int, is_good:bool=True, possible_labels:list=[-1,0,1]):
		self.rtype = rtype
		self.root = root
		self.size = size
		self.id=TreeRule.rule_counter
		self.max_node_id=max_node_id
		self.reversed_cnt=0
		self.is_good=is_good
		self.possible_labels=possible_labels
		TreeRule.rule_counter+=1

	def setsize(self, new_size:int):
		self.size=new_size

	def gen_dot_string(self, comments):
		# color schemes
		reversed_color='yellow'
		added_color='blue'
		reversed_and_added_color='green'
		str_list = ['\n'+comments]
		queue = deque([self.root])
		extra_info = []
		while(queue):
			# print(f"level: {level}, queue: {queue}")
			cur_node = queue.popleft()
			# print(cur_node)
			if(isinstance(cur_node, PredicateNode)):
				if(cur_node.is_added and cur_node.is_reversed):
					color=reversed_and_added_color
					str_list.append(f'{cur_node.number} [color={color}, label="{str(cur_node.pred)}"]')
				elif(cur_node.is_added):
					color=added_color
					str_list.append(f'{cur_node.number} [color={color}, label="{str(cur_node.pred)}"]')
				elif(cur_node.is_reversed):
					color=reversed_color
					str_list.append(f'{cur_node.number} [color={color}, label="{str(cur_node.pred)}"]')
				else:
					str_list.append(f'{cur_node.number} [label="{str(cur_node.pred)}"]')
				str_list.append(f'{cur_node.number}->{cur_node.left.number}')
				str_list.append(f'{cur_node.number}->{cur_node.right.number}')
			else:
				user_inputs = []
				for k,v in cur_node.pairs.items():
					# print('cur_node.pairs')
					# print(cur_node.pairs)
					kstr=f"{str(k)}: ("
					if(self.rtype=='lf'):
						i=1
						for m in v:
							kstr+=f'{str(m.id)} '
							if(i%8==0):
								kstr+='\n'
							i+=1
					elif(self.rtype=='dc'):
						i=1
						for m in v:
							kstr+=f"(t1:{str(m['t1']['_tid_'])}, t2:{str(m['t2']['_tid_'])}) "
							extra_info.append(m['t1']['_tid_'])
							extra_info.append(m['t2']['_tid_'])
							if(i%4==0):
								kstr+='\n'
							i+=1
					kstr+=')\n'

					user_inputs.append(kstr)
					user_input_str="\n".join(user_inputs)
				if(cur_node.is_added and cur_node.is_reversed):
					color=reversed_and_added_color
					str_list.append(f'{cur_node.number} [color={color}, label="{str(cur_node.label)} {user_input_str}"]')
				elif(cur_node.is_added):
					color=added_color
					str_list.append(f'{cur_node.number} [color={color}, label="{str(cur_node.label)} {user_input_str}"]')
				elif(cur_node.is_reversed):
					color=reversed_color
					str_list.append(f'{cur_node.number} [color={color}, label="{str(cur_node.label)} {user_input_str}"]')
				else:
					str_list.append(f'{cur_node.number} [label="{str(cur_node.label)} {user_input_str}"]')
				# str_list.append(f'{cur_node.number} [label="{str(cur_node.label)} {user_input_str}"]')

			if(cur_node.left):
				queue.append(cur_node.left)
			if(cur_node.right):
				queue.append(cur_node.right)
		str_list.append(f"//{','.join([str(x) for x in extra_info])}")
		str_list.append('}')
		# print(f'str_list: {str_list}')
		dot_string= dot_string_template.substitute(nodes_details='\n'.join(str_list))
		return dot_string


	def __str__(self):
		def traverse(node, level):
			indent = '    ' * level
			if isinstance(node, PredicateNode):
				node_info = f"{indent}PredicateNode(id={node.number}, pred={node.pred}"
				if node.is_added:
					node_info += ", added=True"
				if node.is_reversed:
					node_info += ", reversed=True"
				node_info += ")"
			else:
				node_info = f"{indent}LabelNode(id={node.number}, label={node.label}"
				if node.is_added:
					node_info += ", added=True"
				if node.is_reversed:
					node_info += ", reversed=True"
				node_info += ")"

			result.append(node_info)

			if node.left:
				traverse(node.left, level + 1)
			if node.right:
				traverse(node.right, level + 1)

		result = []
		traverse(self.root, 0)
		return f'rid:{self.id}'+'\n'.join(result)
	
	def __repr__(self):
		return self.__str__()

	def evaluate(self, instance: Union[str, dict], ret='label'):
		"""
		return the result given the sentence/ a pair of tuples

		ret: 'label'. 'node'
		"""
		# logger.debug("evaluating rule")
		# logger.debug(self.__str__())
		cur_node=self.root
		# print(f"cur_node: {cur_node}")
		used_predicates = []
		while(cur_node):
			# print(f"cur_node: {cur_node}")
			if(isinstance(cur_node, LabelNode)):
				for u in used_predicates:
					cur_node.used_predicates.add(u)
				if(ret=='label'):
					return cur_node.label
				elif(ret=='node'):
					return cur_node
				else:
					print('invalid return type')
					exit()
				return cur_node
			if(cur_node.pred.evaluate(instance)):
				# print("evaualte to True")
				used_predicates.append(cur_node.pred.pred_identifier)
				cur_node.right.parent=cur_node
				cur_node=cur_node.right
			else:
				cur_node.left.parent=cur_node
				cur_node=cur_node.left


	def gen_label_rule(self, pre=None):
		t=self
		return LabelingFunction(name=self.__str__(), f=self.evaluate, resources={'rule':t}, 
						  use_resourece_as_self=True, 
						  pre=pre, str_repr=self.__str__())


	def serialize(self):
		# return the rule as the acceptyed format for DC/LF Models 
		# LF for Snorkel, Holoclean text format for DC
		res=[]
		if(self.rtype=='dc'):
			
			queue = deque([self.root])

			while(queue):
				cur_node = queue.popleft()
				if(isinstance(cur_node, LabelNode)):
					if(cur_node.label==DIRTY):
						# print("cur_node is dirty. converting the path from root to this dirty label")
						res.append(self.construct_dc_from_dirty(cur_node))
				if(cur_node.left):
					queue.append(cur_node.left)
				if(cur_node.right):
					queue.append(cur_node.right)
		return res

	def construct_dc_from_dirty(self, dirty_node):
		res = ['t1&t2']
		negate_cond=False
		cur_node = dirty_node

		while(cur_node.parent):
			# print(f"cur_node: {cur_node}, cur_node.parent: {cur_node.parent.parent}")
			if(cur_node.parent.left is cur_node):
				if(cur_node.parent.pred.operator=='!='):
					ori_op='IQ'
					replace_op='EQ'
				else:
					ori_op='EQ'
					replace_op='IQ'
				res.append(cur_node.parent.pred.pred.replace(ori_op, replace_op))
			else:
				res.append(cur_node.parent.pred.pred)
			cur_node = cur_node.parent
		# print(f"res: {res}")
		return '&'.join(res)


if __name__ == '__main__':

	x = 'subscribe my channel, I have some good songs'
	# # print(eval("'songs' in x"))
	# # code if song in sentence and sentiment > 0.5 - > HAM 
	# # else abstain
	r1n1 = PredicateNode(pred=KeywordPredicate(keywords=['sons']))
	r1n2 = LabelNode(label=ABSTAIN, used_predicates=set([]))
	r1n3 = PredicateNode(pred=SentimentPredicate(thresh=0.5, sent_func=textblob_sentiment))
	r1n4 = LabelNode(label=ABSTAIN, used_predicates=set([]))
	# r1n4.left=r1n6
	r1n5 = LabelNode(label=HAM, used_predicates=set([]))
	r1n1.left=r1n2
	# r1n1.right=r1n3
	# r1n3.left=r1n4
	r1n1.right=r1n5
	print(r1n1)
	print(r1n2)
	keyword_song_with_sentiment = TreeRule(rtype='lf', root=r1n1, size=3, max_node_id=3)
	# print(keyword_song_with_sentiment)
	print(keyword_song_with_sentiment.evaluate({'text':'i love this song'}))