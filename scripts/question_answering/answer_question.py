import spacy
import itertools
import numpy as np
import torch
import sys
from spacy import displacy
from spacy.symbols import ORTH
from collections import Counter
from collections import defaultdict
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertForQuestionAnswering


YES_NO_WORDS = ["can", "could", "would", "is", "does", "has", "was", "were", "had", "have", "did", "are", "will"]
CHOICE_WORDS = ["or", "either"]
QUESTION_WORDS = ["who", "what", "where", "when", "why", "how", "whose", "which", "whom"]

ENTITY_MAP = defaultdict(lambda: [], 
              {"PERSON": ["PERSON", "ORG"],  # NER sometimes mistakes persons for orgs and vice versa
              "LOCATION": ["FAC", "GPE", "LOC"], # TODO: is this enough? Need to do some more testing
              "DATE": ["DATE", "TIME", "CARDINAL"], # Sometimes labels [year] BC as CARDINAL ORG
              "NUMBER": ["PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"]})

# some sample questions for set 1 doc 1 (Old Kingdom)
S1A1 =  ["What did Sahure command?", 
          "Who did the Fifth Dynasty begin with?", 
          "Who was Sneferu succeeded by?", 
          "Who were the later kings of the Fourth Dynasty?", 
          "Who were the last Pharaohs of the Fifth Dynasty?", 
          "What set in during the reign of Pepi II?", 
          "What followed the collapse of the Old Kingdom?",
          "Who was King Djoser?",
          "Who was Khufu?",
          "What was the Old Kingdom?",
          "Where was the Step Pyramid built?",
          "When did the Old Kingdom reach a zenith?",
          "When was the Fifth Dynasty?",
          "Why was the Sphinx built?",
          "How did ship builders keep their ships assembled?",
          "Was Menkauhor Kaiu one of the last Pharaohs of the Fifth Dynasty?",
          "Was the first Pharaoh of the Old Kingdom Djoser or Snefru?"]


# ============= simple helper functions =============

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


def serve_dep(ex):
  displacy.serve(ex, style="dep")


def stem_and_lower(sent, no_stop=False):
  return [tok.lemma_.lower() for tok in sent if not tok.is_punct and not (no_stop and tok.is_stop)]


def process(sent, no_stop=False):
  return [tok for tok in sent if not tok.is_punct and not (no_stop and tok.is_stop)]


def intersection(a, b):
  return [elem for elem in a if elem in b]


def flatten(nested_list):
  return [elem for sub_list in nested_list for elem in sub_list]




# ============= more complex helper functions =============

def stem_query(query, sent):
  result = []
  query_processed = process(query)
  sent_processed = process(sent)
  sent_toks = list(sent_processed)
  sent_toks_text = [tok.text for tok in sent_toks]
  for tok in query_processed:
    if tok.text in sent_toks_text:
      idx = sent_toks_text.index(tok.text)
      result.append(sent_toks[idx].lemma_.lower())
    else:
      result.append(tok.lemma_.lower())
  return result


def filter_sents_ner(doc, labels, query_ents):
  def has_label(sent):
    sent_labels = [ent.label_ for ent in sent.ents]
    for label in labels:
      if label in sent_labels:
        return True
    return False

  # backup in case an entity was mislabled
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


def lccs(query_tokenized, sent_tokenized):
  mat = np.zeros((len(query_tokenized) + 1, len(sent_tokenized) + 1))
  for i in range(1, len(query_tokenized) + 1):
    for j in range(1, len(sent_tokenized) + 1):
      if query_tokenized[i - 1] == sent_tokenized[j - 1]:
        mat[i][j] = mat[i-1][j-1] + 1
  return np.max(mat)


def ave_dist_sent(query, sent_stem_lower, no_stop=False):
  def get_dist(sent_stem_lower, a, b):
    if a in sent_stem_lower and b in sent_stem_lower:
      last_a = -1
      last_b = -1
      pairs = []
      for i in range(len(sent_stem_lower)):
        if sent_stem_lower[i] == a:
          last_a = i
          if last_b >= 0:
            pairs.append((last_a, last_b))
        elif sent_stem_lower[i] == b:
          last_b = i
          if last_a >= 0:
            pairs.append((last_b, last_a))
      return min([abs(pair[0] - pair[1]) for pair in pairs])
    else:
      return len(sent_stem_lower)

  query_stem_lower = stem_and_lower(query, no_stop)
  query_unique_toks = list(set(query_stem_lower))
  pair_indices = list(itertools.combinations(range(len(query_unique_toks)), 2))
  return sum([get_dist(sent_stem_lower, query_unique_toks[pair[0]], query_unique_toks[pair[1]]) for pair in pair_indices]) / len(pair_indices)


def get_bm25(query, labels, n):
  query_stem_lower = stem_and_lower(query)
  sents_filtered = filter_sents_ner(doc, labels, query.ents)
  queries_stem = [stem_query(query, sent) for sent in sents_filtered]
  sents_stem_lower = [stem_and_lower(sent) for sent in sents_filtered]
  bm25 = BM25Okapi(sents_stem_lower)
  assert(len(queries_stem) == len(sents_filtered))
  scored_sents = [(bm25.get_scores(queries_stem[i])[i], sents_filtered[i]) for i in range(len(sents_filtered))]
  assert(len(scored_sents) == len(sents_filtered))
  #ave_dist_scores = [ave_dist_sent(query, sent_stem_lower) for sent_stem_lower in sents_stem_lower]
  #lccs_scores = [lccs(query_stem_lower, sent_stem_lower) for sent_stem_lower in sents_stem_lower]
  #print(ave_dist_scores)
  #scored_sents = list(zip(bm25.get_scores(query_stem_lower), ave_dist_scores, lccs_scores, sents_filtered))
  scored_sents.sort(key=lambda x: x[0], reverse=True)
  #scored_sents = [(round(entry[0], 1), round(entry[1], 1), entry[2], entry[3]) for entry in scored_sents]
  scored_sents = [(round(entry[0], 1), entry[1]) for entry in scored_sents]
  return scored_sents[:n]



# TODO: add comments
def pattern_match(question_text):
  
  def match_defn12(sent, subj, defn_num, subj_phrase_lower=None):
    # match the first two patterns (x is answer, answer is x)
    # if defn_num is 1, matches first pattern, else matches second pattern
    subj_word = subj.text.lower()
    if defn_num == 1:
      subj_dep = "nsubj"
    else:
      subj_dep = "attr"
    for tok in sent:
      if tok.lemma_ == "be" and tok.pos_ in ["VERB", "AUX"]: # is, are, am, be, etc.
        # get the children of the aux verb in the dependency parse
        tok_children_text = [child.text.lower() for child in list(tok.children)]
        if subj_word in tok_children_text:
          subj_idx = tok_children_text.index(subj_word)
          subj_in_sent = list(tok.children)[subj_idx]
          if subj_in_sent.dep_ == subj_dep:
            if subj_phrase_lower == None:
              return True
            else:
              # Handle multi-word subjects (subj_phrase_lower is not None)
              subj_phrase_lower_in_sent = " ".join(list([tok2.text.lower() for tok2 in subj_in_sent.lefts]) + [subj_in_sent.text.lower()])
              return subj_phrase_lower in subj_phrase_lower_in_sent
    return False
    
    
  def match_defn_appos(sent, subj, subj_phrase_lower=None):
    subj_word = subj.text.lower()
    for tok in sent:
      if tok.text.lower() == subj_word and (tok.pos_ in ["NOUN", "PROPN"]) \
        and (tok.dep_ == "appos" or "appos" in [child.dep_ for child in tok.children]):
        if subj_phrase_lower == None:
          return True
        else:
          # Handle multi-word subjects (subj_phrase_lower is not None)
          subj_phrase_lower_in_sent = " ".join(list([tok2.text.lower() for tok2 in tok.lefts]) + [tok.text.lower()])
          return subj_phrase_lower in subj_phrase_lower_in_sent
      
    return False

  
  def get_matches(subj, subj_phrase_lower=None):
    matches1 = [sent for sent in doc.sents if match_defn12(sent, subj, 1, subj_phrase_lower)]
    matches2 = [sent for sent in doc.sents if match_defn12(sent, subj, 2, subj_phrase_lower)]
    matches_appos = [sent for sent in doc.sents if match_defn_appos(sent, subj, subj_phrase_lower)]
    return [matches1, matches2, matches_appos]
  
  
  question = nlp(question_text)
  question_processed = process(question)
  question_stem_lower = stem_and_lower(question)

  if question_stem_lower[0] in ["who", "what", "where", "when"] and question_stem_lower[1] == "be":
    question_verbs = [tok for tok in question_processed if tok.pos_ in ["VERB", "AUX"]]
    """ question should only have one verb; we want to pattern match "What is a dog", but not
    "What is a dog classified as" (the ranking function can handle the second example) 
    """
    # Special case where we allow "where" questions to have a second verb: "located"
    # (for example, "Where is New York located?")
    where_located_case = (question_stem_lower[0] == "where" and len(question_verbs) == 2 and question_verbs[1].lemma_.lower() == "locate")
    print("where_located_case: {}".format(where_located_case))
    if len(question_verbs) == 1 or where_located_case:
      verb_idx = 0
      if where_located_case:
        verb_idx = 1
      subjs = [tok for tok in list(question_verbs[verb_idx].children) if (tok.dep_ in ["attr", "nsubj"]) \
        and (tok.pos_ in ["NOUN", "PROPN"])]
      print("subjs: " + str(subjs))
      if len(subjs) == 1:
        subj = subjs[0]
        # Make sure the subject is an entity, not a common noun
        # (although should try to handle this case as well)
        if len(nlp("{} is {}?".format(question_processed[0].text, subj)).ents) >= 1:
          if question_stem_lower[0] == "who":
            # "who" question, can just use the single-word subj (likely a name)
            # TODO: what if subj is a last name, and multiple ents in the article share that last name?
            print("Pattern match subj: " + subj.text)
            return get_matches(subj)
          else:
            # not a "who" question, get the whole entity
            qent = question.ents[0]
            subj_phrase_lower = qent.text.lower()
            if subj.text.lower() in subj_phrase_lower:
              if subj.text.lower() == subj_phrase_lower:
                return get_matches(subj)
              print("Pattern match subj phrase: " + subj_phrase_lower)
              return get_matches(subj, subj_phrase_lower)
   
     
  return []





def extract_answer(question_text, sentence_text):
  # question_toks = [tok.text.lower() for tok in nlp(question_text)]
  # sentence_toks = [tok.text.lower() for tok in nlp(sentence_text)]
  # question_ids = bert_tokenizer.convert_tokens_to_ids(question_toks)
  # sentence_ids = bert_tokenizer.convert_tokens_to_ids(sentence_toks)

  print("Question:\n{}".format(question_text))
  print("Sentence:\n{}".format(sentence_text))

  question_ids = bert_tokenizer.encode(question_text)
  sentence_ids = bert_tokenizer.encode(sentence_text)

  input_ids = bert_tokenizer.build_inputs_with_special_tokens(question_ids, sentence_ids)
  input_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)
  print("Tokens:\n{}".format(input_tokens))
  token_type_ids = bert_tokenizer.create_token_type_ids_from_sequences(question_ids, sentence_ids)
  start_logits, end_logits = bert_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

  return " ".join(input_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1]).replace(" ##", "")

  # return bert_tokenizer.convert_ids_to_tokens(input_ids)
  # input_text = "[CLS] " + question_text + " [SEP] " + sentence_text + " [SEP]"
  
  # print("Tokens: " + BertTokenizer.convert_tokens_to_string(input_ids))
  # token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
  # print([(input_ids[i], token_type_ids[i]) for i in range(len(input_ids))])
  # start_logits, end_logits = bert_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
  # input_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)
  
  # return " ".join(input_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1])





def test_extract_answer(idx, sent=None):
  if sent == None:
    sent = get_best_sent(S1A1[idx])
  return extract_answer(S1A1[idx], sent)
  






# ============= testing function to be called by user =============

def get_best_sent(question_text, verbose=True):
  (answer_type, query) = process_question(question_text)
  if verbose:
    print("Question: {}\nQuery: {}\nAnswer type: {}".format(question_text, query, answer_type))
  pattern_matches = flatten(pattern_match(question_text))
  if len(pattern_matches) > 0:
    if verbose:
      print("Retrieval method: pattern match")
    return pattern_matches[0].text.replace("\n", "")
  else:
    if verbose:
      print("Retrieval method: ranking function")
    labels = ENTITY_MAP[answer_type]
    ranked_sents = get_bm25(query, labels, 1)
    if len(ranked_sents) > 0:
      return ranked_sents[0][1].text.replace("\n", "")
    else:
      return list(doc.sents)[0].text.replace("\n", "")

  


if __name__ == '__main__':
  args = sys.argv
  if len(args) >= 3:
    set_num = int(args[1])
    doc_num = int(args[2])
    text = get_text("data/Development_data/set{}/a{}.txt".format(set_num, doc_num))
    nlp = spacy.load("en_core_web_lg")
    sc1 = [{ORTH: "ca."}]
    nlp.tokenizer.add_special_case("ca.", sc1)
    doc = nlp(text)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")




