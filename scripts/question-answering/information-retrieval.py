import spacy
from spacy import displacy
from tabulate import tabulate
from collections import Counter
from collections import defaultdict
import pprint


pp = pprint.PrettyPrinter(indent=4)


def get_text(path="data/Development_data/set1/a1.txt"):
	with open(path, "r") as file:
		text = file.read()
		return text


def tf_idf(sents, query_doc):
	lemmatized_sents = [sent.lemma_.lower() for sent in sents]
	#print(lemmatized_sents)
	num_sents = len(lemmatized_sents)
	idf_dict = defaultdict(lambda: 0)
	for sent in sents:
		token_lemmas = set([token.lemma_.lower() for token in sent])
		for token_lemma in token_lemmas:
			idf_dict[token_lemma] += 1
	pp.pprint(idf_dict)
	sent_tf_idfs = []
	for sent in lemmatized_sents:
		token_tf_idfs = []
		for token in query_doc:
			token_lemma = token.lemma_.lower()
			#print(token_lemma)
			tf = sent.count(token_lemma) / len(sent)
			idf = num_sents / (1 + idf_dict[token_lemma])
			token_tf_idfs.append((tf * idf))
			#print("{}\n{}\n{}, {}\n================".format(token_lemma, sent, tf, idf))
		sent_tf_idfs.append((sent, sum(token_tf_idfs)))

	sent_tf_idfs.sort(key=lambda x: x[1], reverse=True)
	return sent_tf_idfs





def filter_sents(doc, labels, query_ents):
	def has_label(sent):
		sent_labels = [ent.label_ for ent in sent.ents]
		for label in labels:
			if label in sent_labels:
				return True
		print("Filtering out: {}".format(sent))
		return False

	def has_query_ent(sent):
		for ent in query_ents:
			if ent in sent.ents:
				return True
		return False

	return [sent for sent in doc.sents if has_label(sent) or has_query_ent(sent)]


def print_entries(entries):
	for entry in entries:
		print(entry)


def get_entries(query):
	query_doc = nlp(query)
	return tf_idf(filter_sents(doc, labels, query_doc.ents), query_doc)


if __name__ == '__main__':

	nlp = spacy.load("en_core_web_lg")
	text = get_text()
	#query = "King Djoser"
	doc = nlp(text)
	#labels = ["PERSON"]
	#displacy.serve(doc, style="ent")
	


