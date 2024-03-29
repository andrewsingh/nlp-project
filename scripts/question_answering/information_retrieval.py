import spacy
import pprint
import itertools
import numpy as np
from spacy import displacy
from spacy.symbols import ORTH
#from benepar.spacy_plugin import BeneparComponent
from tabulate import tabulate
from collections import Counter
from collections import defaultdict
from rank_bm25 import BM25Okapi


pp = pprint.PrettyPrinter(indent=4)


YES_NO_WORDS = ["can", "could", "would", "is", "does", "has", "was", "were", "had", "have", "did", "are", "will"]
CHOICE_WORDS = ["or", "either"]
QUESTION_WORDS = ["who", "what", "where", "when", "why", "how", "whose", "which", "whom"]

ENTITY_MAP = defaultdict(lambda: [], 
              {"PERSON": ["PERSON", "ORG"],
              "LOCATION": ["FAC", "GPE", "LOC"],
              "DATE": ["DATE", "TIME"],
              "NUMBER": ["PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"]})

S1A1 =  ["What did Sahure command?", 
          "Who did the Fifth Dynasty begin with?", 
          "Who was Sneferu succeeded by?", 
          "Who were the later kings of the Fourth Dynasty?", 
          "Who were the last Pharaohs of the Fifth Dynasty?", 
          "What set in during the reign of Pepi II?", 
          "What followed the collapse of the Old Kingdom?"]


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



def print_list(l):
  for elem in l:
    print(elem)


def print_list2(l):
  for l2 in l:
    for elem in l2:
      print(elem)


def stem_and_lower(sent, no_stop=False):
  return [tok.lemma_.lower() for tok in sent if not tok.is_punct and not (no_stop and tok.is_stop)]


def process(sent, no_stop=False):
  return [tok for tok in sent if not tok.is_punct and not (no_stop and tok.is_stop)]


def intersection(a, b):
  return [elem for elem in a if elem in b]


def flatten(nested_list):
  return [elem for sub_list in nested_list for elem in sub_list]


def filter_sents_ner(doc, labels, query_ents):
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
    return [sent for sent in doc.sents if len(stem_and_lower(sent)) > 0]


def lemmatize_query(query_doc, sent):
  result = []
  query_filtered = process(query_doc)
  sent_filtered = process(sent)
  sent_toks_text = [tok.text for tok in sent]
  sent_toks = list(sent_filtered)
  for tok in query_filtered:
    if tok.text in sent_toks_text:
      idx = sent_toks_text.index(tok.text)
      result.append(sent_toks[idx].lemma_.lower())
    else:
      result.append(tok.lemma_.lower())
  return result


def get_matched_no_stop_words(query_doc, sent):
  query_p = stem_and_lower(query_doc, True)
  sent_p = stem_and_lower(sent, True)
  return len(intersection(query_p, sent_p))


def get_sent(ex):
  ex_doc = nlp(ex)
  return list(ex_doc.sents)[0]


def serve_dep(ex):
  displacy.serve(ex, style="dep")





def process_question(question_text):
  question = nlp(question_text)
  question_word = ""
  question_start = -1

  has_yes_no = False
  answer_type = "NOCATEGORY"
  for (idx, word) in enumerate([tok.text.lower() for tok in question]):
    if word in QUESTION_WORDS:
      question_word = word
      question_start = idx
      break
    elif word in YES_NO_WORDS:
      has_yes_no = True
      if question_start < 0:
        question_start = idx
    elif word in CHOICE_WORDS:
      answer_type = "CHOICE"
      break
  
  if question_start > 0:
    question = question[question_start:] # trim question

  query_toks = process(question) # remove punctuation to get query tokens
  query_word = query_toks[0].text.lower()
  
  if len(question_word) > 0:
    query_toks = query_toks[1:] # remove question word
    if question_word in ["who", "whose", "whom"]:
        answer_type = "PERSON"
    elif question_word in ["what", "which"]:
      if query_word in ["time", "date", "hour", "day", "month", "year", "century"]:
        answer_type = "DATE"
      elif query_word in ["place", "location", "country", "state", "city", "town", "village"]:
        answer_type = "LOCATION"
      elif query_word in ["person", "man", "woman", "boy", "girl"]:
        answer_type = "PERSON"
      elif query_word in ["number", "percent", "percentage", "quantity", "amount", "price"]:
        answer_type = "NUMBER"
    elif question_word == "where":
        answer_type = "LOCATION"
    elif question_word == "when":
        answer_type = "DATE"
    elif question_word == "how":
        if query_word in ["few", "little", "much", "many"]:
            answer_type = "NUMBER"
            query_toks = query_toks[1:]
        elif query_word in ["young", "old", "long"]:
            answer_type = "DATE"
            query_toks = query_toks[1:]
  elif has_yes_no and answer_type != "CHOICE":
    answer_type = "YESNO"

  query = nlp(" ".join([tok.text for tok in query_toks]))
  return (answer_type, query)



def tf_idf(query, labels, n):
  query_doc = nlp(query)
  query_tokenized = [token.lemma_.lower() for token in query_doc]
  sents_filtered = filter_sents_ner(doc, labels, query_doc.ents)
  sents_processed = [stem_and_lower(sent) for sent in sents_filtered]
  sents_tokenized = [sent.split(" ") for sent in sents_processed]

  def tf_idf_sent(sent_tokenized):
    token_tf_idfs = []
    for query_tok in query_tokenized:
      tf = sent_tokenized.count(query_tok) / len(sent_tokenized)
      idf = num_sents / (1 + idf_dict[query_tok])
      token_tf_idfs.append((tf * idf))
      #print("{}\n{}\n{}, {}\n================".format(token_lemma, sent, tf, idf))
    return sum(token_tf_idfs)

  num_sents = len(sents_tokenized)
  idf_dict = defaultdict(lambda: 0)
  for sent_tokenized in sents_tokenized:
    for tok in sent_tokenized:
      idf_dict[tok] += 1
  #pp.pprint(idf_dict)
  sent_tf_idfs = [(tf_idf_sent(sents_tokenized[i]), sents_filtered[i]) for i in range(num_sents)]
  sent_tf_idfs.sort(key=lambda x: x[0], reverse=True)
  return sent_tf_idfs[:n]

    
def lccs(query_tokenized, sent_tokenized):
  mat = np.zeros((len(query_tokenized) + 1, len(sent_tokenized) + 1))
  for i in range(1, len(query_tokenized) + 1):
    for j in range(1, len(sent_tokenized) + 1):
      if query_tokenized[i - 1] == sent_tokenized[j - 1]:
        mat[i][j] = mat[i-1][j-1] + 1
  return np.max(mat)


def ave_dist(query, labels, n, no_stop=False):
  def get_dist(sent_processed, a, b):
    if a in sent_processed and b in sent_processed:
      last_a = -1
      last_b = -1
      pairs = []
      for i in range(len(sent_processed)):
        if sent_processed[i] == a:
          last_a = i
          if last_b >= 0:
            pairs.append((last_a, last_b))
        elif sent_processed[i] == b:
          last_b = i
          if last_a >= 0:
            pairs.append((last_b, last_a))
      return min([abs(pair[0] - pair[1]) for pair in pairs])
    else:
      return len(sent_processed)

  query_stem_lower = stem_and_lower(query, no_stop)
  sents_stem_lower = [stem_and_lower(sent, no_stop) for sent in sents_filtered]
  query_unique_toks = list(set(query_stem_lower))
  pair_indices = list(itertools.combinations(range(len(query_unique_toks)), 2))
  
  def ave_dist_sent(sent_tokenized):
    return sum([get_dist(sent_tokenized, query_unique_toks[pair[0]], query_unique_toks[pair[1]]) for pair in pair_indices]) / len(pair_indices)

  sents_scored = [(ave_dist_sent(sents_stem_lower[i]), sents_filtered[i]) for i in range(len(sents_stem_lower))]
  sents_scored.sort(key=lambda x: x[0])
  return sents_scored[:n]


def get_bm25(query, labels, n):
  query_stem_lower = stem_and_lower(query)
  sents_filtered = filter_sents_ner(doc, labels, query.ents)
  sents_processed = [stem_and_lower(sent) for sent in sents_filtered]
  bm25 = BM25Okapi(sents_processed)
  scored_sents = list(zip(bm25.get_scores(query_stem_lower), sents_filtered))
  scored_sents.sort(key=lambda x: x[0], reverse=True)
  scored_sents = [(round(entry[0], 1), entry[1]) for entry in scored_sents]
  return scored_sents[:n]


def pattern_match(question_text):
  def match_defn12(sent, subj, defn_num):
    subj_word = subj.text.lower()
    if defn_num == 1:
      subj_dep = "nsubj"
    else:
      subj_dep = "attr"
    for tok in sent:
      if tok.lemma_ == "be" and tok.pos_ in ["VERB", "AUX"]:
        tok_children_text = [child.text.lower() for child in list(tok.children)]
        if subj_word in tok_children_text:
          subj_idx = tok_children_text.index(subj_word)
          if list(tok.children)[subj_idx].dep_ == subj_dep:
            return True
    return False
    
  def match_defn_appos(sent, subj):
    subj_word = subj.text.lower()
    for tok in sent:
      if tok.text.lower() == subj_word and (tok.pos_ in ["NOUN", "PROPN"]) \
        and "appos" in [child.dep_ for child in tok.children]:
          return True
    return False

  question = nlp(question_text)
  question_processed = process(question)
  question_stem_lower = stem_and_lower(question)
  if question_stem_lower[0] in ["who", "what"] and question_stem_lower[1] == "be":
    subjs = [tok for tok in list(question_processed[1].children) if (tok.dep_ in ["attr", "nsubj"]) \
              and (tok.pos_ in ["NOUN", "PROPN"])]
    if len(subjs) == 1:
      subj = subjs[0]
      #print("subj: " + subj.text)
      matches1 = [sent for sent in doc.sents if match_defn12(sent, subj, 1)]
      matches2 = [sent for sent in doc.sents if match_defn12(sent, subj, 2)]
      matches_appos = [sent for sent in doc.sents if match_defn_appos(sent, subj)]
      return [matches1, matches2, matches_appos]
    
  return []



def get_best_sent(question_text):
  (answer_type, query) = process_question(question_text)
  pattern_matches = flatten(pattern_match(question_text))
  if len(pattern_matches) > 0:
    return pattern_matches[0].text
  else:
    labels = ENTITY_MAP[answer_type]
    ranked_sents = get_bm25(query, labels, 1)
    if len(ranked_sents) > 0:
      return ranked_sents[0].text
    else:
      return list(doc.sents)[0].text

  


if __name__ == '__main__':

  nlp = spacy.load("en_core_web_lg")
  sc1 = [{ORTH: "ca."}]
  nlp.tokenizer.add_special_case("ca.", sc1)
  #nlp.add_pipe(BeneparComponent("benepar_en2"))
  #nlp2 = spacy.load("en")
  #neuralcoref.add_to_pipe(nlp)
  text = get_text("data/Development_data/set1/a1.txt")
  #question = "Who is King Djoser?"
  doc = nlp(text)
  # query = "The domestic dog is a member of the genus"
  # labels = []
  # query_doc = nlp(query)
  # query_tokenized = [token.lemma_.lower() for token in query_doc]
  # sents_filtered = filter_sents_ner(doc, labels, query_doc.ents)
  # sents_processed = [stem_and_lower(sent) for sent in sents_filtered]
  # q = query_tokenized
  # s = sents_processed[64]
  # qd = query_doc
  # sd = sents_filtered[64]

  # qsm = "The smallest adult dog was"
  # qsm_doc = nlp(qsm)
  # ssm = sents_filtered[77]

  #print(doc._.has_coref)
  #print_entries(doc._.coref_clusters)
  #print(doc._.coref_resolved)


  #reformulations = ["King Djoser was", "was King Djoser"]
  
  #sents_processed = process_sents(doc.sents)
  #bm25 = BM25Okapi(sents)

  #labels = ["PERSON"]
  #displacy.serve(doc, style="ent")
  #displacy.serve(doc, style="dep")
  


