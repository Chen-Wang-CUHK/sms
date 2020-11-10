## Framework for Sentence Mover's Distance

import sys, nltk
import argparse

import numpy as np
import spacy
import math
from wmd import WMD
from nltk.corpus import stopwords
from collections import Counter
from allennlp.commands.elmo import ElmoEmbedder
import json
import scipy.stats as stats

stop_words = set(stopwords.words('english'))

print("loading spacy")
nlp = spacy.load('en_core_web_md')

import pprint
pp = pprint.PrettyPrinter(indent=4)


def print_average_correlation(corr_mat, human_metric, ave_type='macro'):
    corr_mat = np.array(corr_mat)
    results = dict(zip([human_metric + '_' + ave_type + '_kendall', human_metric + '_' + ave_type + '_pearson', human_metric + '_' + ave_type + '_spearman'],
                       [np.mean(corr_mat[:,0]),
                       np.mean(corr_mat[:,1]),
                       np.mean(corr_mat[:,2])]))
    pp.pprint(results)


def tokenize_texts(inLines):

	# input: raw input text
	# output: a list of token IDs, where a id_doc=[[ref],[hyp]],
	#           ref/hyp=[sent1, sent2,...], and a sent=[wordID1, wordID2 ... ]

	id_docs = []
	text_docs = []

	for doc in inLines:
		id_doc = []
		text_doc = []

		for i in range(2):  # iterate over ref and hyp
			text = doc.split('\t')[i].strip()
			sent_list = [sent for sent in nltk.sent_tokenize(text)]
			if WORD_REP == "glove":
				IDs = [[nlp.vocab.strings[t.text.lower()] for t in nlp(sent) if t.text.isalpha() and t.text.lower() not in stop_words] for sent in sent_list]
			if WORD_REP == "elmo":
				# no word IDs, just use spacy ids, but without lower/stop words
				# IDs = [[nlp.vocab.strings[t.text] for t in nlp(sent)] for sent in sent_list]
				IDs = [[nlp.vocab.strings[t.text] for t in nlp(sent)] for sent in sent_list]
			id_list = [x for x in IDs if x != []]  # get rid of empty sents
			text_list = [[token.text for token in nlp(x)] for x in sent_list if x != []]

			id_doc.append(id_list)
			text_doc.append(text_list)
		id_docs.append(id_doc)
		text_docs.append(text_doc)
	return id_docs, text_docs


def get_embeddings(id_doc, text_doc):

	# input: a ref/hyp pair, with each piece is a list of sentences and each sentence is a list of token IDs
	# output: IDs (the orig doc but updating IDs as needed) and rep_map (a dict mapping word ids to embeddings).
	#           if sent emb, add list of sent emb to end of ref and hyp

	rep_map = {}

	# if adding new IDs, make sure they don't overlap with existing IDs
	# to get max, flatten the list of IDs
	new_id = max(sum(sum(id_doc, []), [])) + 1

	sent_ids = [[], []]  # keep track of sentence IDs for rep and hyp. won't use this for wms

	for i in range(2):

		for sent_i in range(len(id_doc[i])):
			sent_emb = []
			word_emb_list = []  # list of a sentence's word embeddings
			# get word embeddings
			if WORD_REP == "glove":
				for wordID in id_doc[i][sent_i]:
					word_emb = nlp.vocab.get_vector(wordID)
					word_emb_list.append(word_emb)
			if WORD_REP == "elmo":
				sent_vec = MODEL.embed_batch([text_doc[i][sent_i]])
				sent_vec = sent_vec[0]  # 1 elt in batch
				word_emb_list = np.average(sent_vec, axis=0)  # average layers to get word embs
				# remove stopwords from elmo
				keep_inds = []
				for word_i in range(len(text_doc[i][sent_i])):
					word = text_doc[i][sent_i][word_i]
					# if the lower-cased word is a stop word or not alphabetic, remove it from emb and id
					if (word.isalpha()) and (word.lower() not in stop_words):
						keep_inds.append(word_i)
				word_emb_list = [word_emb_list[x] for x in range(len(text_doc[i][sent_i])) if x in keep_inds]
				id_doc[i][sent_i] = [id_doc[i][sent_i][x] for x in range(len(text_doc[i][sent_i])) if x in keep_inds]
				assert(len(word_emb_list) == len(id_doc[i][sent_i]))

			# add word embeddings to embedding dict
			if METRIC != "sms":
				for w_ind in range(len(word_emb_list)):
					# if the word is not already in the embedding dict, add it
					w_id = id_doc[i][sent_i][w_ind]
					if w_id not in rep_map:
						rep_map[w_id] = word_emb_list[w_ind]
					# for contextualized embeddings, replace word ID with a unique ID and add it to the embedding dict
					elif WORD_REP != "glove":
						rep_map[new_id] = word_emb_list[w_ind]
						id_doc[i][sent_i][w_ind] = new_id
						new_id += 1

			# add sentence embeddings to embedding dict
			if (METRIC != "wms") and (len(word_emb_list) > 0):
				sent_emb = get_sent_embedding(word_emb_list)
				# add sentence embedding to the embedding dict
				rep_map[new_id] = sent_emb
				sent_ids[i].append(new_id)
				new_id += 1

	# add sentence IDs to ID list
	if METRIC != "wms":
		for j in range(len(id_doc)):
			id_doc[j].append(sent_ids[j])

	return id_doc, rep_map


def get_sent_embedding(emb_list):

	# input: list of a sentence's word embeddings
	# output: the sentence's embedding

	emb_array = np.array(emb_list)
	sent_emb = list(np.mean(emb_array, axis=0))

	return sent_emb


def get_weights(id_doc):

	# input: a ref/hyp pair, with each piece is a list of sentences and each sentence is a list of token IDs.
	#           if the metric is not wms, there is also an extra list of sentence ids for ref and hyp
	# output: 1. a ref/hyp pair of 1-d lists of all word and sentence IDs (where applicable)
	#           2. a ref/hyp pair of arrays of weights for each of those IDs

	# Note that we only need to output counts; these will be normalized by the sum of counts in the WMD code.

	# 2 1-d lists of all relevant embedding IDs
	id_lists = [[], []]
	# 2 arrays where an embedding's weight is at the same index as its ID in id_lists
	d_weights = [np.array([], dtype=np.float32), np.array([], dtype=np.float32)]

	for i in range(len(id_doc)):  # for ref/hyp
		if METRIC != "wms":
			# pop off sent ids so id_doc is back to word ids only
			sent_ids = id_doc[i].pop()

		# collapse to 1-d
		wordIDs = sum(id_doc[i], [])
		# get dict that maps from ID to count
		counts = Counter(wordIDs)

		# get word weights
		if METRIC != "sms":
			for k in counts.keys():
				id_lists[i].append(k)
				d_weights[i] = np.append(d_weights[i], counts[k])

		# get sentence weights
		if METRIC != "wms":
			# weight words by counts and give each sentence a weight equal to the number of words in the sentence
			id_lists[i] += sent_ids
			# make sure to check no empty ids
			d_weights[i] = np.append(d_weights[i], np.array([float(len(x)) for x in id_doc[i] if x != []], dtype=np.float32))

	return id_lists, d_weights


def print_score(inLines, out_file, results_list):

	# input: raw text, the output file, and the results
	# output: scores will be written to output file

	of = open(out_file, 'w')
	of.write("Average: " + str(np.mean(results_list)) + "\n")
	of.write("ID\tReference\tHypothesis\t"+METRIC)
	for i in range(len(inLines)):
		[ref_str, hyp_str] = inLines[i].split('\t')[:2]
		of.write('\n' + str(i) + '\t' + ref_str + '\t' + hyp_str.strip("\n"))
		of.write('\t' + str(results_list[i]))
	of.write('\n')
	of.close()
	return "Done!"


def calc_smd(opts, output_f=""):
	inF = open(opts.input_file, 'r')
	inLines = inF.readlines()
	inF.close()
	print("Found", len(inLines), "documents")
	token_doc_list, text_doc_list = tokenize_texts(inLines)
	count = 0
	results_list = []
	for doc_id in range(len(token_doc_list)):
		doc = token_doc_list[doc_id]
		text = text_doc_list[doc_id]
		# transform doc to ID list, both words and/or sentences. get ID dict that maps to emb
		[ref_ids, hyp_ids], rep_map = get_embeddings(doc, text)
		# get D values
		[ref_id_list, hyp_id_list], [ref_d, hyp_d] = get_weights([ref_ids, hyp_ids])
		# format doc as expected: {id: (id, ref_id_list, ref_d)}
		doc_dict = {"0": ("ref", ref_id_list, ref_d), "1": ("hyp", hyp_id_list, hyp_d)}
		calc = WMD(rep_map, doc_dict, vocabulary_min=1)
		try:
			dist = calc.nearest_neighbors(str(0), k=1, early_stop=1)[0][1]  # how far is hyp from ref?
		except:
			print(doc, text)
		sim = math.exp(-dist)  # switch to similarity
		results_list.append(sim)
		if doc_id == int((len(token_doc_list) / 10.) * count):
			print(str(count * 10) + "% done with calculations")
			count += 1
	# added by wchen to compute correlation scores with human annotated scores
	hscoreF = open(opts.score_file, 'r')
	hscoreLines = hscoreF.readlines()
	hscoreF.close()
	compute_corrs(opts, results_list, hscoreLines)
	# if output_f != "":
	# 	print_score(inLines, output_f, results_list)
	# else:
	# 	print("Results: ", np.mean(results_list))
	# return 'Done!'


def compute_corrs(opts, pred_scores_list, hscoresLines):
	assert len(pred_scores_list) == len(hscoresLines)
	year = opts.input_file.split('tac_')[-1].split('_')[0]
	topic_based_scores = {}
	for idx, line in enumerate(hscoresLines):
		line = line.strip()
		hscores_dict = json.loads(line)
		# add pred_score as the pseudo human score for build the data to compute correlations
		hscores_dict['human_scores']['pred_score'] = pred_scores_list[idx]
		# get topic based scores
		# {"id": "topic006588_doc_name_006588_summ_ml", "human_scores": {"overall": -1.0, "grammar": -0.5, "redundancy": -0.5}}
		topic_summ_id = hscores_dict['id']
		topic = topic_summ_id.split('_')[0][len('topic'):]
		# doc_name = topic_summ_id.split('doc_name_')[-1].split('_')[0]
		sys_name = topic_summ_id.split('summ_')[-1]
		if topic not in topic_based_scores:
			topic_based_scores[topic] = {}
			for metric in hscores_dict['human_scores']:
				topic_based_scores[topic][metric] = {}
				topic_based_scores[topic][metric][sys_name] = [hscores_dict['human_scores'][metric]]
		elif sys_name not in topic_based_scores[topic]['pred_score']:
			for metric in hscores_dict['human_scores']:
				topic_based_scores[topic][metric][sys_name] = [hscores_dict['human_scores'][metric]]
		else:
			for metric in hscores_dict['human_scores']:
				topic_based_scores[topic][metric][sys_name].append(hscores_dict['human_scores'][metric])

	# average the scores from multi-documents for topic-based scores
	for topic in topic_based_scores:
		# to keep the same order
		sys_names_list = list(topic_based_scores[topic]['pred_score'].keys())
		for metric in topic_based_scores[topic]:
			metric_based_scores = []
			for sys_name in sys_names_list:
				topic_based_scores[topic][metric][sys_name] = np.mean(topic_based_scores[topic][metric][sys_name])
				metric_based_scores.append(np.mean(topic_based_scores[topic][metric][sys_name]))
			topic_based_scores[topic][metric] = metric_based_scores

	if year != 'cnndm':
		# {"pyr_score": 0.286, "responsiveness": 2.0}
		target_metrics = ['pyr_score', 'responsiveness']
	else:
		target_metrics = ['overall', 'grammar', 'redundancy']
	# macro averaged corr score
	for trgt_metric in target_metrics:
		topic_based_corr = []
		total_hss = []
		total_pss = []
		for topic in topic_based_scores:
			# skip the case that has equal human scores for each system
			target_scores = topic_based_scores[topic][trgt_metric]
			prediction_scores = topic_based_scores[topic]['pred_score']
			total_hss.extend(target_scores)
			total_pss.extend(prediction_scores)
			if not (np.array(target_scores) == target_scores[0]).all():
				topic_based_corr.append([
					stats.kendalltau(target_scores, prediction_scores)[0],
					stats.pearsonr(target_scores, prediction_scores)[0],
					stats.spearmanr(target_scores, prediction_scores)[0]])
		print('\n====={} Macro RESULTS====='.format(trgt_metric))
		print_average_correlation(topic_based_corr, human_metric=trgt_metric, ave_type='macro')
		# micro averaged correlation scores
		micro_corr = [stats.kendalltau(total_hss, total_pss)[0],
					  stats.pearsonr(total_hss, total_pss)[0],
					  stats.spearmanr(total_hss, total_pss)[0]]
		print('\n====={} Micro RESULTS====='.format(trgt_metric))
		print_average_correlation([micro_corr], human_metric=trgt_metric, ave_type='micro')


if __name__ == "__main__":
	# in_f = sys.argv[1]
	# [WORD_REP, METRIC] = sys.argv[2:4]
	parser = argparse.ArgumentParser(description="smd.py")
	parser.add_argument("--input_file", "-input_file", type=str, default='data/tac_cnndm/sms_input/sms_tac_2010_text_pair.small.txt',
						help="The input data file")
	parser.add_argument("--score_file", "-score_file", type=str,
						default='data/tac_cnndm/sms_input/sms_tac_2010_scores.small.jsonl',
						help="The input data file")
	parser.add_argument("--log_folder", "-log_folder", type=str, default='logs',
						help="The folder that stored the evaluation logs.")
	parser.add_argument("--word_rep", "-word_rep", type=str, default='glove',
						help="The pretrained word representation type.")
	parser.add_argument("--metric", "-metric", type=str, default='s+wms',
						help="The type of evaluation metric.")
	opts = parser.parse_args()

	WORD_REP = opts.word_rep
	METRIC = opts.metric
	word_rep_opt = ["glove", "elmo"]
	metric_opt = ["wms", "sms", "s+wms"]
	if (opts.word_rep not in word_rep_opt) or (opts.metric not in metric_opt):
		raise Exception("Please choose parameters from the following list:\nWORD_REP:\tglove, elmo\n \
		METRIC:\twms, sms, s+wms")
	extension = "_" + opts.word_rep + "_" + opts.metric + ".out"
	out_f = ".".join(opts.input_file.split(".")[:-1]) + extension

	if opts.word_rep == "elmo":
		MODEL = ElmoEmbedder()

	calc_smd(opts, out_f)
