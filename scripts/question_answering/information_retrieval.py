import spacy
import pprint
import itertools
from spacy import displacy
from spacy.symbols import ORTH
#from benepar.spacy_plugin import BeneparComponent
from tabulate import tabulate
from collections import Counter
from collections import defaultdict
from rank_bm25 import BM25Okapi



pp = pprint.PrettyPrinter(indent=4)


def get_text(path="data/Development_data/set1/a1.txt"):
  def trim(lines):
    for i in range(len(lines)):
      line_txt = lines[i].replace("\n", "")
      if line_txt == "See also" or line_txt == "Notes" or line_txt == "References":
        return lines[:i]
    return lines

  with open(path, "r") as file:
    #text = file.read()
    lines = file.readlines()
    sents = [line for line in trim(lines) if "." in line]

  return "".join(sents)


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
  # NER sometimes mistakes persons for orgs and vice versa
  if "PERSON" in labels and "ORG" not in labels:
    labels.append("ORG")
  if "ORG" in labels and "PERSON" not in labels:
    labels.append("PERSON")
  def has_label(sent):
    sent_labels = [ent.label_ for ent in sent.ents]
    for label in labels:
      if label in sent_labels:
        return True
    #print("Filtering out: {}".format(sent))
    return False

  # backup in case an entity was mislabled (e.g., a person was labeled as an org)
  def has_query_ent(sent):
    for ent in query_ents:
      """ as long as an ent in sent.ents has the same text as ent (label can be different), 
        'ent in sent.ents' will be True """
      if ent in sent.ents: 
        return True
    return False

  if len(labels) > 0:
    return [sent for sent in doc.sents if has_label(sent) or has_query_ent(sent)]
  else:
    return list(doc.sents)


def process_sents(sents):
  return [sent.lemma_.lower() for sent in sents]


def print_list(l):
  for elem in l:
    print(elem)



def print_list2(l):
  for l2 in l:
    for elem in l2:
      print(elem)


def get_tf_idf(query, labels):
  query_doc = nlp(query)
  return tf_idf(filter_sents(doc, labels, query_doc.ents), query_doc)


def get_bm25(query, labels, n):
  query_doc = nlp(query)
  query_tokenized = [token.lemma_.lower() for token in query_doc]
  sents_filtered = filter_sents(doc, labels, query_doc.ents)
  sents_processed = process_sents(sents_filtered)
  sents_tokenized = [sent.split(" ") for sent in sents_processed]
  bm25 = BM25Okapi(sents_tokenized)
  scored_sents = list(zip(bm25.get_scores(query_tokenized), sents_filtered))
  scored_sents.sort(key=lambda x: x[0], reverse=True)
  scored_sents = [(round(entry[0], 1), entry[1]) for entry in scored_sents]
  return scored_sents[:n]


def ave_dist(query, labels, n):
  def get_dist(sent_tokenized, a, b):
    if a in sent_tokenized and b in sent_tokenized:
      last_a = -1
      last_b = -1
      pairs = []
      for i in range(len(sent_tokenized)):
        if sent_tokenized[i] == a:
          last_a = i
          if last_b >= 0:
            pairs.append((last_a, last_b))
        elif sent_tokenized[i] == b:
          last_b = i
          if last_a >= 0:
            pairs.append((last_b, last_a))
      return min([abs(pair[0] - pair[1]) for pair in pairs])
    else:
      return len(sent_tokenized)

  query_doc = nlp(query)
  query_tokenized = [token.lemma_.lower() for token in query_doc]
  sents_filtered = filter_sents(doc, labels, query_doc.ents)
  sents_processed = process_sents(sents_filtered)
  sents_tokenized = [sent.split(" ") for sent in sents_processed]
  query_unique_toks = list(set(query_tokenized))
  pair_indices = list(itertools.combinations(range(len(query_unique_toks)), 2))
  
  def ave_dist_sent(sent_tokenized):
    return sum([get_dist(sent_tokenized, query_unique_toks[pair[0]], query_unique_toks[pair[1]]) for pair in pair_indices]) / len(pair_indices)

  sents_scored = [(ave_dist_sent(sents_tokenized[i]), sents_filtered[i]) for i in range(len(sents_tokenized))]
  sents_scored.sort(key=lambda x: x[0])
  return sents_scored[:n]


def pattern_match(question):
  def match_defn12(sent, subj, defn_num):
    if defn_num == 1:
      subj_dep = "nsubj"
    else:
      subj_dep = "attr"
    for tok in sent:
      if tok.lemma_ == "be" and tok.pos_ == "VERB":
        tok_children_text = [child.text for child in list(tok.children)]
        if subj.text in tok_children_text:
          subj_idx = tok_children_text.index(subj.text)
          if list(tok.children)[subj_idx].dep_ == subj_dep:
            return True
    return False
    
  def match_defn_appos(sent, subj):
    for tok in sent:
      if tok.text == subj.text and (tok.pos_ == "NOUN" or tok.pos_ == "PROPN") \
        and "appos" in [child.dep_ for child in tok.children]:
          return True
    return False

  question_doc = nlp(question)
  question_toks = list(question_doc)
  question_toks_p = [token.lemma_.lower() for token in question_doc]
  if (question_toks_p[0] == "who" or question_toks_p[0] == "what") \
    and question_toks_p[1] == "be":

    subjs = [tok for tok in list(question_toks[1].children) if (tok.dep_ == "attr" or tok.dep_ == "nsubj") \
              and (tok.pos_ == "NOUN" or tok.pos_ == "PROPN")]
    if len(subjs) == 1:
      subj = subjs[0]
      matches1 = [sent for sent in doc.sents if match_defn12(sent, subj, 1)]
      matches2 = [sent for sent in doc.sents if match_defn12(sent, subj, 2)]
      matches_appos = [sent for sent in doc.sents if match_defn_appos(sent, subj)]
      return [matches1, matches2, matches_appos]
    else:
      return "Fail 2"

  return "Fail 1"


def get_sent(ex):
  ex_doc = nlp(ex)
  return list(ex_doc.sents)[0]


def serve_dep(ex):
  displacy.serve(ex, style="dep")



if __name__ == '__main__':

  nlp = spacy.load("en_core_web_lg")
  sc1 = [{ORTH: "ca."}]
  nlp.tokenizer.add_special_case("ca.", sc1)
  #nlp.add_pipe(BeneparComponent("benepar_en2"))
  #nlp2 = spacy.load("en")
  #neuralcoref.add_to_pipe(nlp)
  text = get_text("data/Development_data/set5/a1.txt")
  #question = "Who is King Djoser?"
  doc = nlp(text)
  #print(doc._.has_coref)
  #print_entries(doc._.coref_clusters)
  #print(doc._.coref_resolved)


  #reformulations = ["King Djoser was", "was King Djoser"]
  
  #sents_processed = process_sents(doc.sents)
  #bm25 = BM25Okapi(sents)

  #labels = ["PERSON"]
  #displacy.serve(doc, style="ent")
  #displacy.serve(doc, style="dep")
  


