import spacy
from tabulate import tabulate


# path: path to text file
# analysis: 0 for general preprocessing, 1 for named entity recognition
# example: preprocess("../data/Development_data/set1/a1.txt", 1)
def preprocess(path="../data/Development_data/set1/a1.txt", analysis=0):
	with open(path, "r") as file:
		text = file.read()
		nlp = spacy.load("en_core_web_sm")
		doc = nlp(text)
		results = []
		if analysis == 0:
			for token in doc:
				results.append([token.text, token.lemma_, token.pos_, token.tag_, \
					token.dep_, token.shape_, token.is_alpha, token.is_stop])
		elif analysis == 1:
			for ent in doc.ents:
				results.append([ent.text, ent.start_char, ent.end_char, ent.label_])
		print(tabulate(results))
		return



