import sys
import os
sys.path.append(os.path.join(os.getcwd(), ".."))
from rulecleaner_src.TreeRules import (
	KeywordPredicate,
	PredicateNode,
	LabelNode,
	TreeRule,
	SPAM,
	HAM,
	ABSTAIN,
)
from rulecleaner_src.lfs_tree import keyword_labelling_func_builder, regex_func_builder, senti_func_builder, pos_func_builder, length_func_builder


INTER  = 0
SPORTS = 1
TECH = 2
BUSI = 3


def gen_example_funcs():
	TreeRule.rule_counter=0
	f1 = keyword_labelling_func_builder(['songs', 'song'], HAM)
	f2 = keyword_labelling_func_builder(['check'], SPAM)
	f3 = keyword_labelling_func_builder(['love'], HAM)
	f4 = keyword_labelling_func_builder(['shakira'], SPAM)
	f5 = keyword_labelling_func_builder(['checking'], SPAM)
	f6 = regex_func_builder(['http'],SPAM)
	f_sent = senti_func_builder(0.5)
	f_tag=pos_func_builder(['PRPVB'])
	f_length=length_func_builder(5)
	f7 = keyword_labelling_func_builder(['subscribe'], SPAM)

	example_rules = [f1, f2, f3, f4, f5, f6, f_sent, f_tag, f_length, f7]

	return example_rules

def gen_amazon_funcs():
	TreeRule.rule_counter=0
	f1 = keyword_labelling_func_builder(['star', 'stars'], HAM)
	f2 = keyword_labelling_func_builder(['product', 'fit', 'quality', 'size', 'cheap', 'wear'], SPAM)
	f3 = keyword_labelling_func_builder(['great'], HAM)

	# f4 is an estension of f3
	cur_number=1
	tree_size=1
	f4_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['great']))
	cur_number+=1
	tree_size+=1
	f4_root_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f4_root_right_child = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['stars','works']))
	cur_number+=1
	tree_size+=1
	f4_root_right_child_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f4_root_right_child_right_leaf = LabelNode(number=cur_number, label=HAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f4_root.left = f4_root_left_leaf
	f4_root_left_leaf.parent=f4_root
	f4_root.right = f4_root_right_child
	f4_root_right_child.parent=f4_root
	f4_root_right_child.left = f4_root_right_child_left_leaf
	f4_root_right_child_left_leaf.parent=f4_root_right_child
	f4_root_right_child.right = f4_root_right_child_right_leaf
	f4_root_right_child_right_leaf.parent=f4_root_right_child
	f4 = TreeRule(rtype='lf', root=f4_root, size=tree_size, max_node_id=cur_number, is_good=True)

	# f4 = keyword_labelling_func_builder(['great','stars','works'], HAM)
	f5 = keyword_labelling_func_builder(['waste'], SPAM)
	f6 = keyword_labelling_func_builder(['shoes','item','price','comfortable','plastic'], HAM)
	f7 = keyword_labelling_func_builder(['junk','bought','like','dont','just','use','buy','work','small','didnt','did','disappointed'], SPAM)

	# f8 is an estension of f7
	cur_number=1
	tree_size=1
	f8_root = PredicateNode(pred=KeywordPredicate(keywords=['junk','bought','like','dont','just','use','buy','work','small','didnt','did','disappointed']))
	cur_number+=1
	tree_size+=1
	f8_root_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f8_root_right_child = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['shoes','metal','fabric','replace','battery','warranty','plug']))
	cur_number+=1
	tree_size+=1
	f8_root_right_child_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f8_root_right_child_right_leaf = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	
	f8_root.left = f8_root_left_leaf
	f8_root_left_leaf.parent=f8_root

	f8_root.right = f8_root_right_child
	f8_root_right_child.parent=f8_root

	f8_root_right_child.left = f8_root_right_child_left_leaf
	f8_root_right_child_left_leaf.parent=f8_root_right_child
	f8_root_right_child.right = f8_root_right_child_right_leaf
	f8_root_right_child_right_leaf.parent=f8_root_right_child
	f8 = TreeRule(rtype='lf', root=f8_root, size=tree_size, max_node_id=cur_number, is_good=True)


	# f8 = keyword_labelling_func_builder(['junk','bought','like','dont','just','use','buy','work','small','didnt','did','disappointed',
	# 	'shoes','metal','fabric','replace','battery','warranty','plug'], SPAM)

	f9  = keyword_labelling_func_builder(['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy'], HAM)

	# f10 is an estension of f9
	cur_number=1
	tree_size=1
	f10_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy']))
	cur_number+=1
	tree_size+=1
	f10_root_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f10_root_right_child = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['stars', 'soft']))
	cur_number+=1
	tree_size+=1
	f10_root_right_child_left_leaf = LabelNode(number=cur_number, label=ABSTAIN,  pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f10_root_right_child_right_leaf = LabelNode(number=cur_number, label=HAM,  pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f10_root.left = f10_root_left_leaf
	f10_root_left_leaf.parent=f10_root
	f10_root.right = f10_root_right_child
	f10_root_right_child.parent=f10_root
	f10_root_right_child.left = f10_root_right_child_left_leaf
	f10_root_right_child_left_leaf.parent=f10_root_right_child
	f10_root_right_child.right = f10_root_right_child_right_leaf
	f10_root_right_child_right_leaf.parent=f10_root_right_child
	f10 = TreeRule(rtype='lf', root=f10_root,size=tree_size, max_node_id=cur_number, is_good=True)
	# f10  = keyword_labelling_func_builder(['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy', 'stars', 'soft'], HAM)


	# f11 is an estension of f9
	cur_number=1
	tree_size=1
	f11_root = PredicateNode(number=cur_number,pred=KeywordPredicate(keywords=['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy']))
	cur_number+=1
	tree_size+=1
	f11_root_left_leaf = LabelNode(number=cur_number,label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f11_root_right_child = PredicateNode(number=cur_number,pred=KeywordPredicate(keywords=['shoes', 'bought', 'use', 'purchase', 'purchased', 'colors', 'install', 'clean', 'design',
	 'pair', 'screen', 'comfortable', 'products']))
	cur_number+=1
	tree_size+=1
	f11_root_right_child_left_leaf = LabelNode(number=cur_number,label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f11_root_right_child_right_leaf = LabelNode(number=cur_number,label=HAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f11_root.left = f11_root_left_leaf
	f11_root_left_leaf.parent=f11_root
	f11_root.right = f11_root_right_child
	f11_root_right_child.parent=f11_root
	f11_root_right_child.left = f11_root_right_child_left_leaf
	f11_root_right_child_left_leaf.parent=f11_root_right_child
	f11_root_right_child.right = f11_root_right_child_right_leaf
	f11_root_right_child_right_leaf.parent=f11_root_right_child
	f11 = TreeRule(rtype='lf', root=f11_root,size=tree_size, max_node_id=cur_number, is_good=True)

	# f11 = keyword_labelling_func_builder(['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy', 'stars', 'soft',
	# 	'shoes', 'bought', 'use', 'purchase', 'purchased', 'colors', 'install', 'clean', 'design', 'pair', 'screen', 'comfortable', 'products'], HAM)

	f12  = keyword_labelling_func_builder(['returned','broke','battery','cable','fits','install','sturdy','ordered','usb','replacement','brand','installed','unit',
		'batteries','box','warranty','defective','cheaply','durable','advertised'], SPAM)
	f13 = keyword_labelling_func_builder(['cute','shirt'], HAM)
	f14 = keyword_labelling_func_builder(['fabric','return','money','poor','garbage','poorly','terrible','useless','horrible','returning','flimsy'], SPAM)
	f15 = keyword_labelling_func_builder(['pants','looks','toy','color','camera','water','phone','bag','worked','arrived','lasted'], SPAM)

	amazon_rules = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15]
	return amazon_rules


def gen_professor_teacher_funcs():
	# HAM: teacher # SPAM: professor
	# bias_pt
	# \begin{lfList}
	# \item \lfClassB{\textbf{teacher:} students{\lfOr}teacher}\lfStats{(coverage: 22\%, accuracy: 81\%)}
	# \item \lfClassA{\textbf{professor:} research}\lfStats{(coverage: 31\%, accuracy: 92\%)}
	# \item \lfClassB{\textbf{teacher:} husband{\lfOr}loves{\lfOr}lives}\lfStats{(coverage: 7\%, accuracy: 85\%)}
	# \item \lfClassA{\textbf{professor:} ph{\lfOr}phd{\lfOr}postdoctoral}\lfStats{(coverage: 17\%, accuracy: 94\%)}
	# \item \lfClassB{\textbf{teacher:} com{\lfOr}years{\lfOr}music{\lfOr}children{\lfOr}time{\lfOr}classroom{\lfOr}teachers{\lfOr}teach{\lfOr}enjoys{\lfOr}help{\lfOr}schools}\lfStats{(coverage: 39\%, accuracy: 72\%)}
	# \item \lfClassNull{book{\lfOr}press{\lfOr}author{\lfOr}appeared{\lfOr}politics{\lfOr}political}
	# \item \lfClassA{\textbf{professor:} medical{\lfOr}medicine{\lfOr}clinical{\lfOr}health}\lfStats{(coverage: 11\%, accuracy: 86\%)}
	# \item \lfClassNull{people{\lfOr}life{\lfOr}books}
	# \item \lfClassA{\textbf{professor:} engineering{\lfOr}systems{\lfOr}management{\lfOr}computer}\lfStats{(coverage: 12\%, accuracy: 86\%)}
	# \item \lfClassNull{english{\lfOr}literature{\lfOr}writing{\lfOr}art}
	# \end{lfList}
	TreeRule.rule_counter=0
	f1 = keyword_labelling_func_builder(['students', 'teacher'], HAM)
	f2 = keyword_labelling_func_builder(['research'], SPAM)
	f3 = keyword_labelling_func_builder(['husband','loves','lives'], HAM)
	f4 = keyword_labelling_func_builder(['ph','phd','postdoctoral'], SPAM)
	f5 = keyword_labelling_func_builder(['com','years','music','children','time','classroom','teachers','teach','enjoys','help','schools'], HAM)
	f6 = keyword_labelling_func_builder(['medical','medicine','clinical','health'], SPAM)
	f7 = keyword_labelling_func_builder(['engineering','systems','management','computer'], SPAM)
	professor_teacher_funcs = [f1, f2, f3, f4, f5, f6, f7]

	return professor_teacher_funcs

def gen_painter_architecht_funcs():
	# HAM: painter # SPAM: architect
	# bias_pa
	# \begin{lfList}
	# \item \lfClassB{\textbf{painter:} paintings{\lfOr}painting}\lfStats{(coverage: 25\%, accuracy: 99\%)}
	# \item \lfClassA{\textbf{architect:} architecture{\lfOr}development{\lfOr}architect{\lfOr}software{\lfOr}architectural{\lfOr}solutions{\lfOr}management}\lfStats{(coverage: 34\%, accuracy: 95\%)}
	# \item \lfClassB{\textbf{painter:} art{\lfOr}gallery{\lfOr}artist}\lfStats{(coverage: 30\%, accuracy: 88\%)}
	# \item \lfClassA{\textbf{architect:} data{\lfOr}enterprise}\lfStats{(coverage: 5\%, accuracy: 98\%)}
	# \item \lfClassA{\textbf{architect:} experience{\lfOr}projects}\lfStats{(coverage: 19\%, accuracy: 86\%)}
	# \item \lfClassB{\textbf{painter:} paints{\lfOr}life{\lfOr}paint{\lfOr}color}\lfStats{(coverage: 15\%, accuracy: 89\%)}
	# \item \lfClassA{\textbf{architect:} microsoft{\lfOr}technologies{\lfOr}business{\lfOr}services{\lfOr}applications{\lfOr}web{\lfOr}application}\lfStats{(coverage: 13\%, accuracy: 91\%)}
	# \item \lfClassB{\textbf{painter:} york}\lfStats{(coverage: 8\%, accuracy: 71\%)}
	# \item \lfClassB{\textbf{painter:} colors}\lfStats{(coverage: 2\%, accuracy: 97\%)}
	# \item \lfClassB{\textbf{painter:} images{\lfOr}work{\lfOr}arts{\lfOr}born{\lfOr}school{\lfOr}landscape}\lfStats{(coverage: 43\%, accuracy: 70\%)}
	# \end{lfList}
	TreeRule.rule_counter=0
	f1 = keyword_labelling_func_builder(['paintings', 'painting'], HAM)
	f2 = keyword_labelling_func_builder(['architecture','architect','software','architectural','solutions','management'], SPAM)
	f3 = keyword_labelling_func_builder(['art','gallery','artist'], HAM)
	f4 = keyword_labelling_func_builder(['data','enterprise'], SPAM)
	f5 = keyword_labelling_func_builder(['experience','projects'], SPAM)
	f6 = keyword_labelling_func_builder(['microsoft','technologies','business','services','applications','web','application'], SPAM)
	f7 = keyword_labelling_func_builder(['paints','life','paint','color'], HAM)
	f8 = keyword_labelling_func_builder(['york'], HAM)
	f9 = keyword_labelling_func_builder(['colors'], HAM)
	f10 = keyword_labelling_func_builder(['images','work','arts','born','school','landscape'], HAM)

	painter_architecht_funcs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

	return painter_architecht_funcs


def gen_imdb_funcs():
	# HAM: negative, SPAM: positive

	# imdb
	# negative: waste / worst (coverage: 13%, accuracy: 90%)
	# movie
	# 	negative: waste / bad / stupid / crap (coverage: 22%, accuracy: 79%)
	# 	negative: horrible (coverage: 3%, accuracy: 88%)
	# positive: wonderful / excellent / superb (coverage: 14%, accuracy: 80%)
	# positive: loved (coverage: 5%, accuracy: 75%)
	# negative: crap / awful / lame (coverage: 11%, accuracy: 87%)
	# negative: avoid (coverage: 3%, accuracy: 82%)

	TreeRule.rule_counter=0

	cur_number=1
	tree_size=1
	f1 = keyword_labelling_func_builder(['waste', 'worst'], HAM)
	f2_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['movie']))
	cur_number+=1
	tree_size+=1
	f2_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f2_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['waste','bad','stupid','crap']))
	cur_number+=1
	tree_size+=1
	f2_root_right_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f2_root_right_right_leaf = LabelNode(number=cur_number, label=HAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f2_root.left = f2_root_left
	f2_root_left.parent=f2_root
	f2_root.right = f2_root_right
	f2_root_right.parent=f2_root
	f2_root_right.left = f2_root_right_left_leaf
	f2_root_right_left_leaf.parent=f2_root_right
	f2_root_right.right = f2_root_right_right_leaf
	f2_root_right_right_leaf.parent=f2_root_right
	f2 = TreeRule(rtype='lf', root=f2_root, size=tree_size, max_node_id=cur_number, is_good=True)

	cur_number=1
	tree_size=1	
	f3_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['movie']))
	cur_number+=1
	tree_size+=1
	f3_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f3_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['horrible']))
	cur_number+=1
	tree_size+=1
	f3_root_right_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f3_root_right_right_leaf = LabelNode(number=cur_number, label=HAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f3_root.left = f3_root_left
	f3_root_left.parent=f3_root
	f3_root.right = f3_root_right
	f3_root_right.parent=f3_root
	f3_root_right.left = f3_root_right_left_leaf
	f3_root_right_left_leaf.parent=f3_root_right
	f3_root_right.right = f3_root_right_right_leaf
	f3_root_right_right_leaf.parent=f3_root_right
	f3 = TreeRule(rtype='lf', root=f3_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f4 = keyword_labelling_func_builder(['wonderful','excellent', 'superb'], SPAM)
	f5 = keyword_labelling_func_builder(['loved'], SPAM)
	f6 = keyword_labelling_func_builder(['crap','awful', 'lame'], HAM)
	f7 = keyword_labelling_func_builder(['avoid'], HAM)
	imdb_funcs = [f1, f2, f3, f4, f5, f6, f7]

	return imdb_funcs



def gen_pj_funcs():
	# HAM: photographer, SPAM: journalist

	# photographer: photography / images / photographs (coverage: 30%, accuracy: 97%)
	# journalist: journalism / editor / news / reporter / written / radio / writing / reporting / writes / correspondent / journalist / writer / politics / author / twitter (coverage: 40%, accuracy: 89%)
		# journalist: previously / political / public / washington (coverage: 8%, accuracy: 94%)
	# photographer: capture / photos / portraits / camera (coverage: 12%, accuracy: 96%)
	# photographer: style / art / photographer / photographic (coverage: 20%, accuracy: 91%)
	# photographer: make / people / just / beauty / unique (coverage: 14%, accuracy: 71%)
	# photographer: passion (coverage: 4%, accuracy: 82%)
	# photographer: like / fashion / life / pictures (coverage: 15%, accuracy: 71%)
	# journalist: political / health / covered / times / appeared / book / issues / articles / english / daily / previously / tv / blog / weekly / guardian (coverage: 36%, accuracy: 74%)
		# journalist: reporter / written / reporting (coverage: 7%, accuracy: 97%)
	# photographer: clients / work / photo / gallery / portrait / photographing / shooting / light / exhibited / nature / landscape (coverage: 36%, accuracy: 78%)
	# journalist: follow (coverage: 2%, accuracy: 76%)
	# journalist: research / science / business / newspaper / television (coverage: 14%, accuracy: 77%)

	TreeRule.rule_counter=0


	f1 = keyword_labelling_func_builder(['photography', 'images', 'photographs'], HAM)
	f2 = keyword_labelling_func_builder(['journalism', 'editor', 'news', "reporter","written","radio","writing",
									  "reporting","writes","correspondent","journalist","writer","politics","author","twitter",], SPAM)
	
	cur_number=1
	tree_size=1
	f3_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['journalism', 'editor', 'news', "reporter","written","radio","writing",
									  "reporting","writes","correspondent","journalist","writer","politics","author","twitter"]))
	cur_number+=1
	tree_size+=1
	f3_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f3_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["previously", "political", "public", "washington"]))
	cur_number+=1
	tree_size+=1
	f3_root_right_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f3_root_right_right_leaf = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f3_root.left = f3_root_left
	f3_root_left.parent=f3_root
	f3_root.right = f3_root_right
	f3_root_right.parent=f3_root
	f3_root_right.left = f3_root_right_left_leaf
	f3_root_right_left_leaf.parent=f3_root_right
	f3_root_right.right = f3_root_right_right_leaf
	f3_root_right_right_leaf.parent=f3_root_right
	f3 = TreeRule(rtype='lf', root=f3_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f4 = keyword_labelling_func_builder(["capture","photos","portraits","camera"], HAM)
	f5 = keyword_labelling_func_builder(["style","art","photographer","photographic"], HAM)
	f6 = keyword_labelling_func_builder(["make","people","just","beauty","unique"], HAM)
	f7 = keyword_labelling_func_builder(["passion"], HAM)
	f8 = keyword_labelling_func_builder(["like","fashion","life","pictures"], HAM)			 
	f9 = keyword_labelling_func_builder(["political","health","covered","times","appeared","book","issues","articles","english","daily","previously","tv","blog","weekly","guardian"], SPAM)
	
	cur_number=1
	tree_size=1
	f10_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["political","health","covered","times",
																			  "appeared","book","issues","articles","english",
																			  "daily","previously","tv","blog","weekly","guardian"]))
	cur_number+=1
	tree_size+=1
	f10_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f10_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["reporter","written","reporting"]))
	cur_number+=1
	tree_size+=1
	f10_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f10_root_right_right = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f10_root.left = f10_root_left
	f10_root_left.parent=f10_root
	f10_root.right = f10_root_right
	f10_root_right.parent=f10_root
	f10_root_right.left = f10_root_right_left
	f10_root_right_left.parent=f10_root_right
	f10_root_right.right = f10_root_right_right
	f10_root_right_right.parent=f10_root_right
	f10 = TreeRule(rtype='lf', root=f10_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f11 = keyword_labelling_func_builder(["clients","work","photo","gallery","portrait",
									   "photographing","shooting","light","exhibited","nature","landscape"], HAM)
	f12 = keyword_labelling_func_builder(["follow"], SPAM)
	f13 = keyword_labelling_func_builder(["research","science","business","newspaper","television"], SPAM)
	pj_funcs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13]

	return pj_funcs


def gen_pp_funcs():
	# HAM: professor, SPAM: physician

	# physician: specializes (coverage: 24%, accuracy: 95%)
		# physician: spanish (coverage: 15%, accuracy: 100%)
		# physician: center (coverage: 13%, accuracy: 99%)
	# professor: research / ph (coverage: 34%, accuracy: 94%)
	# physician: npi (coverage: 5%, accuracy: 100%)
	# physician: affiliated / medicine (coverage: 38%, accuracy: 92%)
	# physician: practices / english (coverage: 28%, accuracy: 91%)
	# physician: medical / hospital (coverage: 40%, accuracy: 89%)
		# physician: insurance (coverage: 5%, accuracy: 100%)
	# professor: author / history / press / international / journal / social / teaches / project / articles / teaching / courses / theory / political / worked / editor (coverage: 33%, accuracy: 90%)
		# professor: ph / phd (coverage: 10%, accuracy: 99%)
	# physician: number (coverage: 7%, accuracy: 85%)
		# physician: registry (coverage: 3%, accuracy: 100%)
	# physician: family / center / patients / health (coverage: 40%, accuracy: 75%)
	# physician: spanish / shield / carriers / bronze / hmo / speaks / cross / average (coverage: 23%, accuracy: 96%)
	# physician: practicing / credentials (coverage: 5%, accuracy: 95%)
	# professor: engineering / design / holds / politics / development / based / management / media / works / data / recent / business / master / professor / public / technology / environmental / systems / study (coverage: 40%, accuracy: 80%)
		# professor: ph / phd (coverage: 12%, accuracy: 99%)

	TreeRule.rule_counter=0

	f1 = keyword_labelling_func_builder(['specializes'], SPAM)

	cur_number=1
	tree_size=1
	f2_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['specializes']))
	cur_number+=1
	tree_size+=1
	f2_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f2_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["spanish"]))
	cur_number+=1
	tree_size+=1
	f2_root_right_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f2_root_right_right_leaf = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f2_root.left = f2_root_left
	f2_root_left.parent=f2_root
	f2_root.right = f2_root_right
	f2_root_right.parent=f2_root
	f2_root_right.left = f2_root_right_left_leaf
	f2_root_right_left_leaf.parent=f2_root_right
	f2_root_right.right = f2_root_right_right_leaf
	f2_root_right_right_leaf.parent=f2_root_right
	f2 = TreeRule(rtype='lf', root=f2_root, size=tree_size, max_node_id=cur_number, is_good=True)

	
	cur_number=1
	tree_size=1
	f3_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['specializes']))
	cur_number+=1
	tree_size+=1
	f3_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f3_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["center"]))
	cur_number+=1
	tree_size+=1
	f3_root_right_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f3_root_right_right_leaf = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f3_root.left = f3_root_left
	f3_root_left.parent=f3_root
	f3_root.right = f3_root_right
	f3_root_right.parent=f3_root
	f3_root_right.left = f3_root_right_left_leaf
	f3_root_right_left_leaf.parent=f3_root_right
	f3_root_right.right = f3_root_right_right_leaf
	f3_root_right_right_leaf.parent=f3_root_right
	f3 = TreeRule(rtype='lf', root=f3_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f4 = keyword_labelling_func_builder(["research", "ph"], HAM)
	f5 = keyword_labelling_func_builder(["npi"], SPAM)
	f6 = keyword_labelling_func_builder(["affiliated", "medicine"], SPAM)
	f7 = keyword_labelling_func_builder(["practices", "english"], SPAM)
	f8 = keyword_labelling_func_builder(["medical","hospital"], SPAM)


	cur_number=1
	tree_size=1
	f9_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["medical","hospital"]))
	cur_number+=1
	tree_size+=1
	f9_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f9_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["insurance"]))
	cur_number+=1
	tree_size+=1
	f9_root_right_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f9_root_right_right_leaf = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f9_root.left = f9_root_left
	f9_root_left.parent=f9_root
	f9_root.right = f9_root_right
	f9_root_right.parent=f9_root
	f9_root_right.left = f9_root_right_left_leaf
	f9_root_right_left_leaf.parent=f9_root_right
	f9_root_right.right = f9_root_right_right_leaf
	f9_root_right_right_leaf.parent=f9_root_right
	f9 = TreeRule(rtype='lf', root=f9_root, size=tree_size, max_node_id=cur_number, is_good=True)


	f10 = keyword_labelling_func_builder(["author","history","press","international","journal","social","teaches","project","articles","teaching","courses","theory","political","worked","editor"], SPAM)
	

	cur_number=1
	tree_size=1
	f11_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["author","history","press","international","journal","social",
																			 "teaches","project","articles","teaching","courses","theory","political","worked","editor"]))
	cur_number+=1
	tree_size+=1
	f11_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f11_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["ph","phd"]))
	cur_number+=1
	tree_size+=1
	f11_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f11_root_right_right = LabelNode(number=cur_number, label=HAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f11_root.left = f11_root_left
	f11_root_left.parent=f11_root
	f11_root.right = f11_root_right
	f11_root_right.parent=f11_root
	f11_root_right.left = f11_root_right_left
	f11_root_right_left.parent=f11_root_right
	f11_root_right.right = f11_root_right_right
	f11_root_right_right.parent=f11_root_right
	f11 = TreeRule(rtype='lf', root=f11_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f12 = keyword_labelling_func_builder(["number"], SPAM)

	cur_number=1
	tree_size=1
	f13_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["number"]))
	cur_number+=1
	tree_size+=1
	f13_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f13_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["registry"]))
	cur_number+=1
	tree_size+=1
	f13_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f13_root_right_right = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f13_root.left = f13_root_left
	f13_root_left.parent=f13_root
	f13_root.right = f13_root_right
	f13_root_right.parent=f13_root
	f13_root_right.left = f13_root_right_left
	f13_root_right_left.parent=f13_root_right
	f13_root_right.right = f13_root_right_right
	f13_root_right_right.parent=f13_root_right
	f13 = TreeRule(rtype='lf', root=f13_root, size=tree_size, max_node_id=cur_number, is_good=True)


	f14 = keyword_labelling_func_builder(["family","center","patients","health"], SPAM)
	f15 = keyword_labelling_func_builder(["spanish","shield","carriers","bronze","hmo","speaks","cross","average"], SPAM)
	f16 = keyword_labelling_func_builder(["practicing","credentials"], SPAM)
	f17 = keyword_labelling_func_builder(["engineering","design","holds","politics","development","based","management","media","works","data","recent","business","master",
									   "professor","public","technology","environmental","systems","study"], HAM)
	
	cur_number=1
	tree_size=1
	f18_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["engineering","design","holds","politics",
									   "development","based","management","media","works","data","recent","business","master",
									   "professor","public","technology","environmental","systems","study"]))
	cur_number+=1
	tree_size+=1
	f18_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f18_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["ph","phd"]))
	cur_number+=1
	tree_size+=1
	f18_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f18_root_right_right = LabelNode(number=cur_number, label=HAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f18_root.left = f18_root_left
	f18_root_left.parent=f18_root
	f18_root.right = f18_root_right
	f18_root_right.parent=f18_root
	f18_root_right.left = f18_root_right_left
	f18_root_right_left.parent=f18_root_right
	f18_root_right.right = f18_root_right_right
	f18_root_right_right.parent=f18_root_right
	f18 = TreeRule(rtype='lf', root=f18_root, size=tree_size, max_node_id=cur_number, is_good=True)

	pp_funcs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17,f18]

	return pp_funcs



def gen_yelp_funcs():
	# HAM: negative, SPAM: positive

	# bad: customer / rude / horrible / worst / terrible / company (coverage: 19%, accuracy: 84%)
	# good: delicious (coverage: 8%, accuracy: 86%)
	# good: love / great / amazing / favorite (coverage: 41%, accuracy: 71%)
	# bad: phone / called (coverage: 8%, accuracy: 74%)
	# good: excellent (coverage: 4%, accuracy: 82%)
	# bad: dirty (coverage: 2%, accuracy: 89%)
	# good: helpful (coverage: 3%, accuracy: 76%)
	# bad: employees (coverage: 2%, accuracy: 71%)


	TreeRule.rule_counter=0

	f1 = keyword_labelling_func_builder(["customer","rude","horrible","worst","terrible","company"], HAM)
	f2 = keyword_labelling_func_builder(["delicious"], SPAM)
	f3 = keyword_labelling_func_builder(["love","great","amazing","favorite"], SPAM)
	f4 = keyword_labelling_func_builder(["phone","called"], HAM)
	f5 = keyword_labelling_func_builder(["excellent"], SPAM)
	f6 = keyword_labelling_func_builder(["dirty"], HAM)
	f7 = keyword_labelling_func_builder(["helpful"], SPAM)
	f8 = keyword_labelling_func_builder(["employees"], HAM)
	
	yelp_funcs = [f1, f2, f3, f4, f5, f6, f7, f8]

	return yelp_funcs

def gen_plots_funcs():
	# HAM: action, SPAM: romance

	# action: world / fight / war / agent (coverage: 14%, accuracy: 81%)
	# romance: woman (coverage: 10%, accuracy: 84%)
	# romance: marriage / student / mysterious (coverage: 7%, accuracy: 74%)
	# romance: falls (coverage: 4%, accuracy: 89%)
	# romance: couple / life / family / daughter / ex / town / friend / years / way / guy (coverage: 32%, accuracy: 72%)
	# romance: relationship / york (coverage: 7%, accuracy: 90%)
	# romance: story (coverage: 5%, accuracy: 84%)
	# romance: takes (coverage: 2%, accuracy: 73%)
	# romance: girl (coverage: 3%, accuracy: 87%)
	# romance: friends (coverage: 4%, accuracy: 84%)

	TreeRule.rule_counter=0

	f1 = keyword_labelling_func_builder(["world","fight","war","agent"], HAM)
	f2 = keyword_labelling_func_builder(["woman"], SPAM)
	f3 = keyword_labelling_func_builder(["marriage","student","mysterious"], SPAM)
	f4 = keyword_labelling_func_builder(["fails"], SPAM)
	f5 = keyword_labelling_func_builder(["couple","life","family","daughter","ex","town","friend","years","way","guy"], SPAM)
	f6 = keyword_labelling_func_builder(["relationship", "york"], SPAM)
	f7 = keyword_labelling_func_builder(["story"], SPAM)
	f8 = keyword_labelling_func_builder(["takes"], SPAM)
	f9 = keyword_labelling_func_builder(["girl"], SPAM)
	f10 = keyword_labelling_func_builder(["friends"], SPAM)
	plot_funcs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

	return plot_funcs

def gen_fakenews_funcs():
	# HAM: true, SPAM: fake

	# fake: "video","featured" (coverage: 36%, accuracy: 96%)
		# fake: "getty","screenshot" (coverage: 10%, accuracy: 100%)
	# true: "minister","ministry","parliament" (coverage: 16%, accuracy: 89%)
	# fake: "pic" (coverage: 8%, accuracy: 100%)
	# true: "wednesday","spokesman","thursday","representatives","nov" (coverage: 39%, accuracy: 75%)
		# true: "legislation" (coverage: 4%, accuracy: 87%)
	# true: "korea","missile","region","regional","authorities" (coverage: 16%, accuracy: 78%)
		# true: "korean" (coverage: 2%, accuracy: 86%)
	# fake: "getty","watch","image","com","https","don","woman","didn","gop" (coverage: 46%, accuracy: 88%)
	# true: "talks" (coverage: 6%, accuracy: 85%)
	# true: "rex" (coverage: 2%, accuracy: 83%)
	# true: "northern","turkey","britain","forces","ruling","european" (coverage: 20%, accuracy: 78%)

	TreeRule.rule_counter=0

	f1 = keyword_labelling_func_builder(["video","featured"], SPAM)
	
	cur_number=1
	tree_size=1
	f2_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["getty","screenshot"]))
	cur_number+=1
	tree_size+=1
	f2_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f2_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["registry"]))
	cur_number+=1
	tree_size+=1
	f2_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f2_root_right_right = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f2_root.left = f2_root_left
	f2_root_left.parent=f2_root
	f2_root.right = f2_root_right
	f2_root_right.parent=f2_root
	f2_root_right.left = f2_root_right_left
	f2_root_right_left.parent=f2_root_right
	f2_root_right.right = f2_root_right_right
	f2_root_right_right.parent=f2_root_right
	f2 = TreeRule(rtype='lf', root=f2_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f3 = keyword_labelling_func_builder(["minister","ministry","parliament"], HAM)

	f4 = keyword_labelling_func_builder(["pic"], SPAM)
	f5 = keyword_labelling_func_builder(["wednesday","spokesman","thursday","representatives","nov"], HAM)

	cur_number=1
	tree_size=1
	f5_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["wednesday","spokesman","thursday","representatives","nov"]))
	cur_number+=1
	tree_size+=1
	f5_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f5_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["legislation"]))
	cur_number+=1
	tree_size+=1
	f5_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f5_root_right_right = LabelNode(number=cur_number, label=HAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f5_root.left = f5_root_left
	f5_root_left.parent=f5_root
	f5_root.right = f5_root_right
	f5_root_right.parent=f5_root
	f5_root_right.left = f5_root_right_left
	f5_root_right_left.parent=f5_root_right
	f5_root_right.right = f5_root_right_right
	f5_root_right_right.parent=f5_root_right
	f5 = TreeRule(rtype='lf', root=f5_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f6 = keyword_labelling_func_builder(["korea","missile","region","regional","authorities"], HAM)

	cur_number=1
	tree_size=1
	f7_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["korea","missile","region","regional","authorities"]))
	cur_number+=1
	tree_size+=1
	f7_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f7_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["korean"]))
	cur_number+=1
	tree_size+=1
	f7_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f7_root_right_right = LabelNode(number=cur_number, label=HAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f7_root.left = f7_root_left
	f7_root_left.parent=f7_root
	f7_root.right = f7_root_right
	f7_root_right.parent=f7_root
	f7_root_right.left = f7_root_right_left
	f7_root_right_left.parent=f7_root_right
	f7_root_right.right = f7_root_right_right
	f7_root_right_right.parent=f7_root_right
	f7 = TreeRule(rtype='lf', root=f7_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f8 = keyword_labelling_func_builder(["getty","watch","image","com","https","don","woman","didn","gop"], SPAM)
	f9 = keyword_labelling_func_builder(["talks"], HAM)
	f10 = keyword_labelling_func_builder(["rex"], SPAM)
	f11 = keyword_labelling_func_builder(["northern","turkey","britain","forces","ruling","european"], HAM)

	fake_news_funcs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]

	return fake_news_funcs


def gen_dbpedia_funcs():
	# HAM: company, SPAM: politics

	# Company: company / based / founded / headquartered / owned / products / label (coverage: 42%, accuracy: 97%)
		# Company: airline / services / software / manufacturer / provides / operates / production / limited (coverage: 17%, accuracy: 100%)
	# Politics: politician / born / member / served (coverage: 45%, accuracy: 97%)
		# Politics: representatives (coverage: 8%, accuracy: 100%)
		# Politics: canadian (coverage: 3%, accuracy: 99%)
		# Politics: assembly / legislative (coverage: 6%, accuracy: 100%)
	# Politics: minister / constituency (coverage: 10%, accuracy: 99%)
	# Politics: representatives / district / state (coverage: 20%, accuracy: 90%)
		# Politics: representing (coverage: 4%, accuracy: 100%)
	# Company: services / located / largest / established / group / international / corporation / provides / service / operates / manufacturer / companies / headquarters / airline / world (coverage: 36%, accuracy: 89%)
	# Politics: american / republican (coverage: 16%, accuracy: 71%)
	# Company: record / records / software / known (coverage: 14%, accuracy: 79%)
		# Company: label (coverage: 4%, accuracy: 100%)
	# Politics: house / governor / january / general / april / august / july / president / november / february / september / october / march / december / june / appointed / mayor / university / county / party / secretary (coverage: 56%, accuracy: 82%)
	# Company: chinese / bank / limited / london / independent / chain / brand / main / technology / private / music (coverage: 23%, accuracy: 85%)
	# Politics: assembly / legislative / law / serving / national / government / political / lawyer / leader / representative / court / elected (coverage: 31%, accuracy: 85%)
	# Company: california / york / management / business / including / major / firm / subsidiary / production / offices / operated (coverage: 25%, accuracy: 81%)
	# Politics: war / son / william / john / senate / school / democratic / senator / term / college / representing / represented / council / chief / 14 / 24 / 10 / 12 / 22 / 21 / 17 / 15 / 11 / 16 / election / 28 (coverage: 43%, accuracy: 85%)


	TreeRule.rule_counter=0

	f1 = keyword_labelling_func_builder(["company","based","founded","headquartered","owned","products","label"], HAM)
	
	cur_number=1
	tree_size=1
	f2_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["company","based","founded","headquartered","owned","products","label"]))
	cur_number+=1
	tree_size+=1
	f2_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f2_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["airline","services","software","manufacturer","provides","operates","production","limited"]))
	cur_number+=1
	tree_size+=1
	f2_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f2_root_right_right = LabelNode(number=cur_number, label=HAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f2_root.left = f2_root_left
	f2_root_left.parent=f2_root
	f2_root.right = f2_root_right
	f2_root_right.parent=f2_root
	f2_root_right.left = f2_root_right_left
	f2_root_right_left.parent=f2_root_right
	f2_root_right.right = f2_root_right_right
	f2_root_right_right.parent=f2_root_right
	f2 = TreeRule(rtype='lf', root=f2_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f3 = keyword_labelling_func_builder(["politician","born","member","served"], SPAM)

	cur_number=1
	tree_size=1
	f4_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["politician","born","member","served"]))
	cur_number+=1
	tree_size+=1
	f4_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f4_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["representatives"]))
	cur_number+=1
	tree_size+=1
	f4_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f4_root_right_right = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f4_root.left = f4_root_left
	f4_root_left.parent=f4_root
	f4_root.right = f4_root_right
	f4_root_right.parent=f4_root
	f4_root_right.left = f4_root_right_left
	f4_root_right_left.parent=f4_root_right
	f4_root_right.right = f4_root_right_right
	f4_root_right_right.parent=f4_root_right
	f4 = TreeRule(rtype='lf', root=f4_root, size=tree_size, max_node_id=cur_number, is_good=True)


	cur_number=1
	tree_size=1
	f5_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["politician","born","member","served"]))
	cur_number+=1
	tree_size+=1
	f5_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f5_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["canadian"]))
	cur_number+=1
	tree_size+=1
	f5_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f5_root_right_right = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f5_root.left = f5_root_left
	f5_root_left.parent=f5_root
	f5_root.right = f5_root_right
	f5_root_right.parent=f5_root
	f5_root_right.left = f5_root_right_left
	f5_root_right_left.parent=f5_root_right
	f5_root_right.right = f5_root_right_right
	f5_root_right_right.parent=f5_root_right
	f5 = TreeRule(rtype='lf', root=f5_root, size=tree_size, max_node_id=cur_number, is_good=True)


	cur_number=1
	tree_size=1
	f6_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["politician","born","member","served"]))
	cur_number+=1
	tree_size+=1
	f6_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f6_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["assembly", "legislative"]))
	cur_number+=1
	tree_size+=1
	f6_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f6_root_right_right = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f6_root.left = f6_root_left
	f6_root_left.parent=f6_root
	f6_root.right = f6_root_right
	f6_root_right.parent=f6_root
	f6_root_right.left = f6_root_right_left
	f6_root_right_left.parent=f6_root_right
	f6_root_right.right = f6_root_right_right
	f6_root_right_right.parent=f6_root_right
	f6 = TreeRule(rtype='lf', root=f6_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f7 = keyword_labelling_func_builder(["minister","constituency"], SPAM)

	f8 = keyword_labelling_func_builder(["representatives","district", "state"], SPAM)


	cur_number=1
	tree_size=1
	f9_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["representatives","district", "state"]))
	cur_number+=1
	tree_size+=1
	f9_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f9_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["representing"]))
	cur_number+=1
	tree_size+=1
	f9_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f9_root_right_right = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f9_root.left = f9_root_left
	f9_root_left.parent=f9_root
	f9_root.right = f9_root_right
	f9_root_right.parent=f9_root
	f9_root_right.left = f9_root_right_left
	f9_root_right_left.parent=f9_root_right
	f9_root_right.right = f9_root_right_right
	f9_root_right_right.parent=f9_root_right
	f9 = TreeRule(rtype='lf', root=f9_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f10 = keyword_labelling_func_builder(["services","located","largest","established","group","international","corporation","provides","service",
									  "operates","manufacturer","companies","headquarters","airline","world"], HAM)
	
	f11 = keyword_labelling_func_builder(["american", "republican"], SPAM)

	f12 = keyword_labelling_func_builder(["record","records","software","known"], HAM)
	
	cur_number=1
	tree_size=1
	f13_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["record","records","software","known"]))
	cur_number+=1
	tree_size+=1
	f13_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f13_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["label"]))
	cur_number+=1
	tree_size+=1
	f13_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f13_root_right_right = LabelNode(number=cur_number, label=HAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f13_root.left = f13_root_left
	f13_root_left.parent=f13_root
	f13_root.right = f13_root_right
	f13_root_right.parent=f13_root
	f13_root_right.left = f13_root_right_left
	f13_root_right_left.parent=f13_root_right
	f13_root_right.right = f13_root_right_right
	f13_root_right_right.parent=f13_root_right
	f13 = TreeRule(rtype='lf', root=f13_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f14 = keyword_labelling_func_builder(["house","governor","january","general","april","august","july","president",
									   "november","february","september","october","march","december","june","appointed",
									   "mayor","university","county","party","secretary"], SPAM)
	
	f15 = keyword_labelling_func_builder(["chinese","bank","limited","london","independent","chain","brand","main","technology","private","music"], HAM)
	f16 = keyword_labelling_func_builder(["assembly","legislative","law","serving","national","government","political","lawyer","leader","representative","court","elected"], SPAM)
	f17 = keyword_labelling_func_builder(["california","york","management","business","including","major","firm","subsidiary","production","offices","operated"], HAM)
	f18 = keyword_labelling_func_builder(["war","son","william","john","senate","school","democratic","senator","term","college","representing","represented",\
									   "council","chief","14","24","10","12","22","21","17","15","11","16","election","28"], SPAM)

	dbpedia_funcs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18]

	return dbpedia_funcs




def gen_agnews_funcs():
	# HAM: Technology, SPAM: Business

	# binary_agnews
	# Technology: space / microsoft / announced / software / users / windows (coverage: 19%, accuracy: 82%)
	# Business: oil (coverage: 6%, accuracy: 97%)
	# Business: york (coverage: 6%, accuracy: 89%)
	# Business: aspx (coverage: 2%, accuracy: 100%)
	# Technology: music / internet / web / service / online / quot / security (coverage: 20%, accuracy: 77%)
	# Business: economy / prices / bank (coverage: 9%, accuracy: 93%)
	# Business: quarter / profit / percent / sales / growth / earnings (coverage: 15%, accuracy: 82%)
	# Technology: ibm / wireless (coverage: 5%, accuracy: 75%)
	# Business: dollar / investors / federal / government (coverage: 10%, accuracy: 78%)

	TreeRule.rule_counter=0

	f1 = keyword_labelling_func_builder(["space","microsoft","announced","software","users","windows"], HAM)
	f2 = keyword_labelling_func_builder(["oil"], SPAM)
	f3 = keyword_labelling_func_builder(["york"], SPAM)
	f4 = keyword_labelling_func_builder(["aspx"], SPAM)
	f5 = keyword_labelling_func_builder(["music","internet","web","service","online","quot","security"], HAM)
	f6 = keyword_labelling_func_builder(["economy","prices","bank"], SPAM)
	f7 = keyword_labelling_func_builder(["quarter", "profit", "percent", "sales", "growth", "earnings"], SPAM)
	f8 = keyword_labelling_func_builder(["ibm","wireless"], HAM)
	f9 = keyword_labelling_func_builder(["dollar","investors","federal","government"], SPAM)

	agnews_funcs = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

	return agnews_funcs


def gen_tweets_funcs():
	# HAM: Positive, SPAM: Negative

	# # airline_tweets
	# negative: united (coverage: 27%, accuracy: 84%)
	# negative: usairways / southwestair / americanair (coverage: 59%, accuracy: 82%)
	# positive: southwestair (coverage: 3%, accuracy: 78%)
	# negative: flightled (coverage: 4%, accuracy: 98%)
	# negative: hold / phone (coverage: 8%, accuracy: 98%)
	# 	negative: usairways (coverage: 3%, accuracy: 99%)
	# negative: bag / plane (coverage: 8%, accuracy: 92%)
	# negative: customer (coverage: 6%, accuracy: 84%)
	# negative: flight (coverage: 24%, accuracy: 88%)
	# 	negative: southwestair (coverage: 4%, accuracy: 79%)
	# 	negative: americanair / usairways (coverage: 11%, accuracy: 94%)
	# negative: bags (coverage: 2%, accuracy: 97%)
	# negative: airline (coverage: 3%, accuracy: 79%)
	# negative: crew / delayed (coverage: 6%, accuracy: 86%)
	# negative: people / don / worst / waiting (coverage: 10%, accuracy: 95%)
	# 	negative: americanair (coverage: 2%, accuracy: 97%)
	# 	negative: usairways / united (coverage: 6%, accuracy: 97%)

	TreeRule.rule_counter=0

	f1 = keyword_labelling_func_builder(["united"], SPAM)
	f2 = keyword_labelling_func_builder(["usairways","southwestair","americanair"], SPAM)
	f3 = keyword_labelling_func_builder(["southwestair"], HAM)
	f4 = keyword_labelling_func_builder(["flightled"], SPAM)
	f5 = keyword_labelling_func_builder(["hold", "phone"], HAM)

	cur_number=1
	tree_size=1
	f6_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["hold", "phone"]))
	cur_number+=1
	tree_size+=1
	f6_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f6_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["usairways"]))
	cur_number+=1
	tree_size+=1
	f6_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f6_root_right_right = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f6_root.left = f6_root_left
	f6_root_left.parent=f6_root
	f6_root.right = f6_root_right
	f6_root_right.parent=f6_root
	f6_root_right.left = f6_root_right_left
	f6_root_right_left.parent=f6_root_right
	f6_root_right.right = f6_root_right_right
	f6_root_right_right.parent=f6_root_right
	f6 = TreeRule(rtype='lf', root=f6_root, size=tree_size, max_node_id=cur_number, is_good=True)
	

	f7 = keyword_labelling_func_builder(["bag","plane"], SPAM)
	f8 = keyword_labelling_func_builder(["customer"], SPAM)
	f8 = keyword_labelling_func_builder(["flight"], SPAM)


	cur_number=1
	tree_size=1
	f9_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["flight"]))
	cur_number+=1
	tree_size+=1
	f9_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f9_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["southwestair"]))
	cur_number+=1
	tree_size+=1
	f9_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f9_root_right_right = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f9_root.left = f9_root_left
	f9_root_left.parent=f9_root
	f9_root.right = f9_root_right
	f9_root_right.parent=f9_root
	f9_root_right.left = f9_root_right_left
	f9_root_right_left.parent=f9_root_right
	f9_root_right.right = f9_root_right_right
	f9_root_right_right.parent=f9_root_right
	f9 = TreeRule(rtype='lf', root=f9_root, size=tree_size, max_node_id=cur_number, is_good=True)


	cur_number=1
	tree_size=1
	f10_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["flight"]))
	cur_number+=1
	tree_size+=1
	f10_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f10_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["americanair","usairways"]))
	cur_number+=1
	tree_size+=1
	f10_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f10_root_right_right = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f10_root.left = f10_root_left
	f10_root_left.parent=f10_root
	f10_root.right = f10_root_right
	f10_root_right.parent=f10_root
	f10_root_right.left = f10_root_right_left
	f10_root_right_left.parent=f10_root_right
	f10_root_right.right = f10_root_right_right
	f10_root_right_right.parent=f10_root_right
	f10 = TreeRule(rtype='lf', root=f10_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f11 = keyword_labelling_func_builder(["bags"], SPAM)
	f12 = keyword_labelling_func_builder(["airline"], SPAM)
	f13 = keyword_labelling_func_builder(["crew", "delayed"], SPAM)

	f14 = keyword_labelling_func_builder(["people","don","worst","waiting"], SPAM)


	cur_number=1
	tree_size=1
	f15_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["people","don","worst","waiting"]))
	cur_number+=1
	tree_size+=1
	f15_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f15_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["americanair"]))
	cur_number+=1
	tree_size+=1
	f15_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f15_root_right_right = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f15_root.left = f15_root_left
	f15_root_left.parent=f15_root
	f15_root.right = f15_root_right
	f15_root_right.parent=f15_root
	f15_root_right.left = f15_root_right_left
	f15_root_right_left.parent=f15_root_right
	f15_root_right.right = f15_root_right_right
	f15_root_right_right.parent=f15_root_right
	f15 = TreeRule(rtype='lf', root=f15_root, size=tree_size, max_node_id=cur_number, is_good=True)

	cur_number=1
	tree_size=1
	f16_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["people","don","worst","waiting"]))
	cur_number+=1
	tree_size+=1
	f16_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f16_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["usairways", "united"]))
	cur_number+=1
	tree_size+=1
	f16_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f16_root_right_right = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f16_root.left = f16_root_left
	f16_root_left.parent=f16_root
	f16_root.right = f16_root_right
	f16_root_right.parent=f16_root
	f16_root_right.left = f16_root_right_left
	f16_root_right_left.parent=f16_root_right
	f16_root_right.right = f16_root_right_right
	f16_root_right_right.parent=f16_root_right
	f16 = TreeRule(rtype='lf', root=f16_root, size=tree_size, max_node_id=cur_number, is_good=True)

	tweets_funcs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16]

	return tweets_funcs



def gen_spam_funcs():

	# spam
	# ham: lor / going / da (coverage: 8%, accuracy: 99%)
	# spam: txt / www / free (coverage: 7%, accuracy: 81%)
	# ham: ok (coverage: 5%, accuracy: 99%)
	# ham: sorry (coverage: 3%, accuracy: 99%)
	# ham: love (coverage: 3%, accuracy: 93%)
	# ham: lt (coverage: 4%, accuracy: 100%)
	# ham: good (coverage: 4%, accuracy: 95%)
	# ham: oh / come (coverage: 6%, accuracy: 99%)
	# ham: did (coverage: 2%, accuracy: 98%)
	# ham: number / dont / just / new / day (coverage: 16%, accuracy: 76%)
	# ham: don (coverage: 2%, accuracy: 97%)
	# ham: ll / home / later / need (coverage: 12%, accuracy: 98%)
	# ham: know (coverage: 4%, accuracy: 92%)
	# ham: time (coverage: 3%, accuracy: 91%)
	# ham: hey / hi (coverage: 4%, accuracy: 89%)
	# ham: like (coverage: 4%, accuracy: 92%)
	# ham: got (coverage: 3%, accuracy: 96%)

	TreeRule.rule_counter=0

	f1 = keyword_labelling_func_builder(["lor", "going", "da"], HAM)
	f2 = keyword_labelling_func_builder(["txt","www","free"], SPAM)
	f3 = keyword_labelling_func_builder(["ok"], HAM)
	f4 = keyword_labelling_func_builder(["sorry"], HAM)
	f5 = keyword_labelling_func_builder(["love"], HAM)
	f6 = keyword_labelling_func_builder(["lt"], HAM)
	f7 = keyword_labelling_func_builder(["good"], HAM)
	f8 = keyword_labelling_func_builder(["oh","come"], HAM)

	f9 = keyword_labelling_func_builder(["did"], HAM)

	f10 = keyword_labelling_func_builder(["number", "dont", "just", "new", "day"], HAM)
	f11 = keyword_labelling_func_builder(["don"], HAM)
	f12 = keyword_labelling_func_builder(["ll","home", "later", "need"], HAM)
	f13 = keyword_labelling_func_builder(["know"], HAM)
	f14 = keyword_labelling_func_builder(["time"], HAM)
	f15 = keyword_labelling_func_builder(["hey",'hi'], HAM)
	f16 = keyword_labelling_func_builder(['like'], HAM)
	f17 = keyword_labelling_func_builder(["got"], HAM) 

	spam_funcs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17]

	return spam_funcs



def gen_gpt_refined_amazon_funcs_user_input_20():
	TreeRule.rule_counter=0
	f1 = keyword_labelling_func_builder(['star', 'stars', 'rating', 'out of five', 'rated'], HAM)
	f2 = keyword_labelling_func_builder(['product', 'fit', 'quality', 'size', 'cheap', 'wear', 'material', 'design', 'durable'], SPAM)
	f3 = keyword_labelling_func_builder(['great'], HAM)

	# f4 is an estension of f3
	cur_number=1
	tree_size=1
	f4_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['great', 'excellent', 'amazing']))
	cur_number+=1
	tree_size+=1
	f4_root_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f4_root_right_child = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['stars','works','fantastic', 'love']))
	cur_number+=1
	tree_size+=1
	f4_root_right_child_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f4_root_right_child_right_leaf = LabelNode(number=cur_number, label=HAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f4_root.left = f4_root_left_leaf
	f4_root_left_leaf.parent=f4_root
	f4_root.right = f4_root_right_child
	f4_root_right_child.parent=f4_root
	f4_root_right_child.left = f4_root_right_child_left_leaf
	f4_root_right_child_left_leaf.parent=f4_root_right_child
	f4_root_right_child.right = f4_root_right_child_right_leaf
	f4_root_right_child_right_leaf.parent=f4_root_right_child
	f4 = TreeRule(rtype='lf', root=f4_root, size=tree_size, max_node_id=cur_number, is_good=True)

	# f4 = keyword_labelling_func_builder(['great','stars','works'], HAM)
	f5 = keyword_labelling_func_builder(['waste','not worth', 'disappointed', 'regret', 'poor value'], SPAM)
	f6 = keyword_labelling_func_builder(['shoes','item','price','comfortable','plastic'], HAM)
	f7 = keyword_labelling_func_builder(['junk', 'bought', 'like', 'dont', 'just', 'use', 'buy', 'work', 'small', 'didnt', 'did', 'disappointed', 
                'bad', 'terrible', 'horrible', 'awful', 'useless'], SPAM)

	# f8 is an estension of f7
	cur_number=1
	tree_size=1
	f8_root = PredicateNode(pred=KeywordPredicate(keywords=['junk', 'bought', 'like', 'dont', 'just', 'use', 'buy', 'work', 'small', 'didnt', 'did', 'disappointed', 
                 'bad', 'terrible', 'horrible', 'awful', 'useless']))
	cur_number+=1
	tree_size+=1
	f8_root_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f8_root_right_child = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['shoes', 'metal', 'fabric', 
																						'replace', 'battery', 'warranty', 'plug', 'defective', 'broken']))
	cur_number+=1
	tree_size+=1
	f8_root_right_child_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f8_root_right_child_right_leaf = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	
	f8_root.left = f8_root_left_leaf
	f8_root_left_leaf.parent=f8_root

	f8_root.right = f8_root_right_child
	f8_root_right_child.parent=f8_root

	f8_root_right_child.left = f8_root_right_child_left_leaf
	f8_root_right_child_left_leaf.parent=f8_root_right_child
	f8_root_right_child.right = f8_root_right_child_right_leaf
	f8_root_right_child_right_leaf.parent=f8_root_right_child
	f8 = TreeRule(rtype='lf', root=f8_root, size=tree_size, max_node_id=cur_number, is_good=True)


	# f8 = keyword_labelling_func_builder(['junk','bought','like','dont','just','use','buy','work','small','didnt','did','disappointed',
	# 	'shoes','metal','fabric','replace','battery','warranty','plug'], SPAM)

	f9  = keyword_labelling_func_builder(['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy', 'fantastic', 'recommend'], HAM)

	# f10 is an estension of f9
	cur_number=1
	tree_size=1
	f10_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy']))
	cur_number+=1
	tree_size+=1
	f10_root_left_leaf = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f10_root_right_child = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=['stars', 'soft']))
	cur_number+=1
	tree_size+=1
	f10_root_right_child_left_leaf = LabelNode(number=cur_number, label=ABSTAIN,  pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f10_root_right_child_right_leaf = LabelNode(number=cur_number, label=HAM,  pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f10_root.left = f10_root_left_leaf
	f10_root_left_leaf.parent=f10_root
	f10_root.right = f10_root_right_child
	f10_root_right_child.parent=f10_root
	f10_root_right_child.left = f10_root_right_child_left_leaf
	f10_root_right_child_left_leaf.parent=f10_root_right_child
	f10_root_right_child.right = f10_root_right_child_right_leaf
	f10_root_right_child_right_leaf.parent=f10_root_right_child
	f10 = TreeRule(rtype='lf', root=f10_root,size=tree_size, max_node_id=cur_number, is_good=True)
	# f10  = keyword_labelling_func_builder(['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy', 'stars', 'soft'], HAM)


	# f11 is an estension of f9
	cur_number=1
	tree_size=1
	f11_root = PredicateNode(number=cur_number,pred=KeywordPredicate(keywords=['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy']))
	cur_number+=1
	tree_size+=1
	f11_root_left_leaf = LabelNode(number=cur_number,label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f11_root_right_child = PredicateNode(number=cur_number,pred=KeywordPredicate(keywords=['shoes', 'bought', 'use', 'purchase', 'purchased', 'colors', 'install', 'clean', 'design',
	 'pair', 'screen', 'comfortable', 'products']))
	cur_number+=1
	tree_size+=1
	f11_root_right_child_left_leaf = LabelNode(number=cur_number,label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f11_root_right_child_right_leaf = LabelNode(number=cur_number,label=HAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f11_root.left = f11_root_left_leaf
	f11_root_left_leaf.parent=f11_root
	f11_root.right = f11_root_right_child
	f11_root_right_child.parent=f11_root
	f11_root_right_child.left = f11_root_right_child_left_leaf
	f11_root_right_child_left_leaf.parent=f11_root_right_child
	f11_root_right_child.right = f11_root_right_child_right_leaf
	f11_root_right_child_right_leaf.parent=f11_root_right_child
	f11 = TreeRule(rtype='lf', root=f11_root,size=tree_size, max_node_id=cur_number, is_good=True)

	# f11 = keyword_labelling_func_builder(['love', 'perfect', 'loved', 'nice', 'excellent', 'works', 'loves', 'awesome', 'easy', 'stars', 'soft',
	# 	'shoes', 'bought', 'use', 'purchase', 'purchased', 'colors', 'install', 'clean', 'design', 'pair', 'screen', 'comfortable', 'products'], HAM)

	f12  = keyword_labelling_func_builder(['returned','broke','battery','cable','fits','install','sturdy','ordered','usb','replacement','brand','installed','unit',
		'batteries','box','warranty','defective','cheaply','durable','advertised'], SPAM)
	f13 = keyword_labelling_func_builder(['cute','shirt'], HAM)
	f14 = keyword_labelling_func_builder(['fabric', 'return', 'money', 'poor', 'garbage', 'poorly', 'terrible', 'useless', 'horrible', 'returning', 'flimsy', 'falling apart'], SPAM)
	f15 = keyword_labelling_func_builder(['pants','looks','toy','color','camera','water','phone','bag','worked','arrived','lasted'], SPAM)

	amazon_rules = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15]
	return amazon_rules


def gen_gpt_refined_fakenews_funcs_user_input_40():
	# HAM: true, SPAM: fake

	# fake: "video","featured" (coverage: 36%, accuracy: 96%)
		# fake: "getty","screenshot" (coverage: 10%, accuracy: 100%)
	# true: "minister","ministry","parliament" (coverage: 16%, accuracy: 89%)
	# fake: "pic" (coverage: 8%, accuracy: 100%)
	# true: "wednesday","spokesman","thursday","representatives","nov" (coverage: 39%, accuracy: 75%)
		# true: "legislation" (coverage: 4%, accuracy: 87%)
	# true: "korea","missile","region","regional","authorities" (coverage: 16%, accuracy: 78%)
		# true: "korean" (coverage: 2%, accuracy: 86%)
	# fake: "getty","watch","image","com","https","don","woman","didn","gop" (coverage: 46%, accuracy: 88%)
	# true: "talks" (coverage: 6%, accuracy: 85%)
	# true: "rex" (coverage: 2%, accuracy: 83%)
	# true: "northern","turkey","britain","forces","ruling","european" (coverage: 20%, accuracy: 78%)

	TreeRule.rule_counter=0

	f1 = keyword_labelling_func_builder(["video", "featured", "clip", "footage"], SPAM)
	cur_number=1
	tree_size=1
	f2_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["getty", "screenshot", "image"]))
	cur_number+=1
	tree_size+=1
	f2_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f2_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["registry", "database"]))
	cur_number+=1
	tree_size+=1
	f2_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f2_root_right_right = LabelNode(number=cur_number, label=SPAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f2_root.left = f2_root_left
	f2_root_left.parent=f2_root
	f2_root.right = f2_root_right
	f2_root_right.parent=f2_root
	f2_root_right.left = f2_root_right_left
	f2_root_right_left.parent=f2_root_right
	f2_root_right.right = f2_root_right_right
	f2_root_right_right.parent=f2_root_right
	f2 = TreeRule(rtype='lf', root=f2_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f3 = keyword_labelling_func_builder(["minister", "ministry", "parliament", "government", "official"], HAM)

	f4 = keyword_labelling_func_builder(["pic", "photo", "image", "snapshot"], SPAM)
	f5 = keyword_labelling_func_builder(["wednesday", "spokesman", "thursday", "representatives", "nov", "tuesday", "monday"], HAM)

	cur_number=1
	tree_size=1
	f5_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["wednesday", "spokesman", "thursday", "representatives", "nov", "tuesday", "monday"]))
	cur_number+=1
	tree_size+=1
	f5_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f5_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["legislation", "bill", "law"]))
	cur_number+=1
	tree_size+=1
	f5_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f5_root_right_right = LabelNode(number=cur_number, label=HAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f5_root.left = f5_root_left
	f5_root_left.parent=f5_root
	f5_root.right = f5_root_right
	f5_root_right.parent=f5_root
	f5_root_right.left = f5_root_right_left
	f5_root_right_left.parent=f5_root_right
	f5_root_right.right = f5_root_right_right
	f5_root_right_right.parent=f5_root_right
	f5 = TreeRule(rtype='lf', root=f5_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f6 = keyword_labelling_func_builder(["korea", "missile", "region", "regional", "authorities", "conflict", "war", "border"], HAM)

	cur_number=1
	tree_size=1
	f7_root = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["korea", "missile", "region", "regional", "authorities", "conflict", "war", "border"]))
	cur_number+=1
	tree_size+=1
	f7_root_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f7_root_right = PredicateNode(number=cur_number, pred=KeywordPredicate(keywords=["korean", "diplomatic"]))
	cur_number+=1
	tree_size+=1
	f7_root_right_left = LabelNode(number=cur_number, label=ABSTAIN, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	cur_number+=1
	tree_size+=1
	f7_root_right_right = LabelNode(number=cur_number, label=HAM, pairs={ABSTAIN:[], SPAM:[], HAM:[]}, used_predicates=set([]))
	f7_root.left = f7_root_left
	f7_root_left.parent=f7_root
	f7_root.right = f7_root_right
	f7_root_right.parent=f7_root
	f7_root_right.left = f7_root_right_left
	f7_root_right_left.parent=f7_root_right
	f7_root_right.right = f7_root_right_right
	f7_root_right_right.parent=f7_root_right
	f7 = TreeRule(rtype='lf', root=f7_root, size=tree_size, max_node_id=cur_number, is_good=True)

	f8 = keyword_labelling_func_builder(["getty", "watch", "image", "com", "https", "don", "woman", "didn", "gop", "site", "web"], SPAM)
	f9 = keyword_labelling_func_builder(["talks", "negotiations", "discussions"], HAM)
	f10 = keyword_labelling_func_builder(["rex", "rex images", "rex news"], SPAM)
	f11 = keyword_labelling_func_builder(["northern", "turkey", "britain", "forces", "ruling", "european", "alliance", "nato", "eu"], HAM)

	fake_news_funcs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]

	return fake_news_funcs


def gen_agnews_funcs_4_class():
	
	#### 0: international
	r1 = ["atomic", "captives", "baghdad", "israeli", "iraqis", "iranian", "afghanistan", "wounding", "terrorism", "soldiers", \
    "palestinians", "palestinian", "policemen", "iraqi", "terrorist", 'north korea', 'korea', \
    'israel', 'u.n.', 'egypt', 'iran', 'iraq', 'nato', 'armed', 'peace']
	f1 = keyword_labelling_func_builder(r1, INTER, possible_labels=[0,1,2,3])
	
	r2= [' war ', 'prime minister', 'president', 'commander', 'minister',  'annan', "military", "militant", "kill", 'operator']
	
	f2 = keyword_labelling_func_builder(r2, INTER, possible_labels=[0,1,2,3])

    #### 1: sports
	r3 = ["goals", "bledsoe", "coaches",  "touchdowns", "kansas", "rankings", "no.", \
     "champ", "cricketers", "hockey", "champions", "quarterback", 'club', 'team',  'baseball', 'basketball', 'soccer', 'football', 'boxing',  'swimming', \
     'world cup', 'nba',"olympics","final", "finals", 'fifa',  'racist', 'racism']
	f3 = keyword_labelling_func_builder(r3, SPORTS, possible_labels=[0,1,2,3])
	
	r4 = ['athlete',  'striker', 'defender', 'goalkeeper',  'midfielder', 'shooting guard', 'power forward', 'point guard', 'pitcher', 'catcher', 'first base', 'second base', 'third base','shortstop','fielder']
	f4 = keyword_labelling_func_builder(r4, SPORTS, possible_labels=[0,1,2,3])

    #r3 = ["tech", "digital", "internet", "mobile"]
	r5=['lakers','chelsea', 'piston','cavaliers', 'rockets', 'clippers','ronaldo', \
        'celtics', 'hawks','76ers', 'raptors', 'pacers', 'suns', 'warriors','blazers','knicks','timberwolves', 'hornets', 'wizards', 'nuggets', 'mavericks', 'grizzlies', 'spurs', \
        'cowboys', 'redskins', 'falcons', 'panthers', 'eagles', 'saints', 'buccaneers', '49ers', 'cardinals', 'texans', 'seahawks', 'vikings', 'patriots', 'colts', 'jaguars', 'raiders', 'chargers', 'bengals', 'steelers', 'browns', \
        'braves','marlins','mets','phillies','cubs','brewers','cardinals', 'diamondbacks','rockies', 'dodgers', 'padres', 'orioles', 'sox', 'yankees', 'jays', 'sox', 'indians', 'tigers', 'royals', 'twins','astros', 'angels', 'athletics', 'mariners', 'rangers', \
        'arsenal', 'burnley', 'newcastle', 'leicester', 'manchester united', 'everton', 'southampton', 'hotspur','tottenham', 'fulham', 'watford', 'sheffield','crystal palace', 'derby', 'charlton', 'aston villa', 'blackburn', 'west ham', 'birmingham city', 'middlesbrough', \
        'real madrid', 'barcelona', 'villarreal', 'valencia', 'betis', 'espanyol','levante', 'sevilla', 'juventus', 'inter milan', 'ac milan', 'as roma', 'benfica', 'porto', 'getafe', 'bayern', 'schalke', 'bremen', 'lyon', 'paris saint', 'monaco', 'dynamo']
	f5 = keyword_labelling_func_builder(r5, SPORTS, possible_labels=[0,1,2,3])
    
	#### 3:tech
	r6 = ["technology", "engineering", "science", "research", "cpu", "windows", "unix", "system", 'computing',  'compute']#, "wireless","chip", "pc", ]
	f6 = keyword_labelling_func_builder(r6, TECH, possible_labels=[0,1,2,3])

	r7= ["google", "apple", "microsoft", "nasa", "yahoo", "intel", "dell", \
    'huawei',"ibm", "siemens", "nokia", "samsung", 'panasonic', \
     't-mobile', 'nvidia', 'adobe', 'salesforce', 'linkedin', 'silicon', 'wiki'
    ]
	f7 = keyword_labelling_func_builder(r7, TECH,  possible_labels=[0,1,2,3])

    #### 2:business
	r8= ["stock", "account", "financ", "goods", "retail", 'economy', 'chairman', 'bank', 'deposit', 'economic', 'dow jones', 'index', '$',  'percent', 'interest rate', 'growth', 'profit', 'tax', 'loan',  'credit', 'invest']
	f8 = keyword_labelling_func_builder(r8, BUSI, possible_labels=[0,1,2,3])

	r9= ["delta", "cola", "toyota", "costco", "gucci", 'citibank', 'airlines']
	f9 = keyword_labelling_func_builder(r9, BUSI, possible_labels=[0,1,2,3])

	agnews_4_calss_funcs = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

	return agnews_4_calss_funcs


def gen_chemprot_func():
      
      ## Part of
	f1 = keyword_labelling_func_builder(['amino acid'], 0, possible_labels=list(range(0,10)))
	f2 = keyword_labelling_func_builder(['replace'], 0, possible_labels=list(range(0,10)))
	f3 = keyword_labelling_func_builder(['mutant', 'mutat'],0, possible_labels=list(range(0,10)))
      
# ## Part of
# @labeling_function()
# def lf_amino_acid(x):
#     return 1 if 'amino acid' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_replace(x):
#     return 1 if 'replace' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_mutant(x):
#     return 1 if 'mutant' in x.sentence.lower() or 'mutat' in x.sentence.lower() else ABSTAIN

	f4 = keyword_labelling_func_builder(['bind'], 1, possible_labels=list(range(0,10)))
	f5 = keyword_labelling_func_builder(['interact'], 1, possible_labels=list(range(0,10)))
	f6 = keyword_labelling_func_builder(['affinit'],1, possible_labels=list(range(0,10)))
# ## Regulator
# @labeling_function()
# def lf_bind(x):
#     return 2 if 'bind' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_interact(x):
#     return 2 if 'interact' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_affinity(x):
#     return 2 if 'affinit' in x.sentence.lower() else ABSTAIN

	f7 = keyword_labelling_func_builder(['activat'], 2, possible_labels=list(range(0,10)))
	f8 = keyword_labelling_func_builder(['increas'], 2, possible_labels=list(range(0,10)))
	f9 = keyword_labelling_func_builder(['induc'], 2, possible_labels=list(range(0,10)))
	f10 = keyword_labelling_func_builder(['stimulat'],2, possible_labels=list(range(0,10)))
	f11 = keyword_labelling_func_builder(['upregulat'],2, possible_labels=list(range(0,10)))

## Upregulator
# Activator
# @labeling_function()
# def lf_activate(x):
#     return 3 if 'activat' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_increase(x):
#     return 3 if 'increas' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_induce(x):
#     return 3 if 'induc' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_stimulate(x):
#     return 3 if 'stimulat' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_upregulate(x):
#     return 3 if 'upregulat' in x.sentence.lower() else ABSTAIN


	f12 = keyword_labelling_func_builder(['downregulat', 'down-regulat'], 3, possible_labels=list(range(0,10)))
	f13 = keyword_labelling_func_builder(['reduc'], 3, possible_labels=list(range(0,10)))
	f14 = keyword_labelling_func_builder(['inhibit'],3, possible_labels=list(range(0,10)))
	f15 = keyword_labelling_func_builder(['decreas'],3, possible_labels=list(range(0,10)))

## Downregulator
# @labeling_function()
# def lf_downregulate(x):
#     return 4 if 'downregulat' in x.sentence.lower() or 'down-regulat' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_reduce(x):
#     return 4 if 'reduc' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_inhibit(x):
#     return 4 if 'inhibit' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_decrease(x):
#     return 4 if 'decreas' in x.sentence.lower() else ABSTAIN


	f16 = keyword_labelling_func_builder([' agoni', '\tagoni'], 4, possible_labels=list(range(0,10)))

## Agonist
# @labeling_function()
# def lf_agonist(x):
#     return 5 if ' agoni' in x.sentence.lower() or "\tagoni" in x.sentence.lower() else ABSTAIN

	f17 = keyword_labelling_func_builder(['antagon'], 5, possible_labels=list(range(0,10)))

## Antagonist
# @labeling_function() 
# def lf_antagonist(x):
#     return 6 if 'antagon' in x.sentence.lower() else ABSTAIN
	f18 = keyword_labelling_func_builder(['modulat'], 6, possible_labels=list(range(0,10)))
	f19 = keyword_labelling_func_builder(['allosteric'], 6, possible_labels=list(range(0,10)))

## Modulator
# @labeling_function()
# def lf_modulate(x):
#     return 7 if 'modulat' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_allosteric(x):
#     return 7 if 'allosteric' in x.sentence.lower() else ABSTAIN

	f20 = keyword_labelling_func_builder(['cofactor'], 7, possible_labels=list(range(0,10)))

## Cofactor
# @labeling_function()
# def lf_cofactor(x):
#     return 8 if 'cofactor' in x.sentence.lower() else ABSTAIN

	f21 = keyword_labelling_func_builder(['substrate'], 8, possible_labels=list(range(0,10)))
	f22 = keyword_labelling_func_builder(['transport'], 8, possible_labels=list(range(0,10)))
	f23 = keyword_labelling_func_builder(['catalyz', 'catalys'], 8, possible_labels=list(range(0,10)))
	f24 = keyword_labelling_func_builder(['produc'], 8, possible_labels=list(range(0,10)))
	f25 = keyword_labelling_func_builder(['conver'], 8, possible_labels=list(range(0,10)))
## Substrate/Product
# @labeling_function()
# def lf_substrate(x):
#     return 9 if 'substrate' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_transport(x):
#     return 9 if 'transport' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_catalyze(x):
#     return 9 if 'catalyz' in x.sentence.lower() or 'catalys' in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_product(x):
#     return 9 if "produc" in x.sentence.lower() else ABSTAIN

# @labeling_function()
# def lf_convert(x):
#     return 9 if "conver" in x.sentence.lower() else ABSTAIN
	f26 = keyword_labelling_func_builder(['not'], 9, possible_labels=list(range(0,10)))

## NOT
# @labeling_function()
# def lf_not(x):
#     return 10 if 'not' in x.sentence.lower() else ABSTAIN
	return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16,
		 f17, f18, f19, f20, f21, f22, f23, f24, f25, f26]