import spacy
import itertools
import numpy as np
import torch
import sys
import re
import warnings
from spacy import displacy
from spacy.symbols import ORTH
from collections import Counter
from collections import defaultdict
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertForQuestionAnswering


warnings.filterwarnings("ignore")




YES_NO_WORDS = ["am", "can", "could", "would", "is", "does", "has", "was", "were", "had",                 "have", "did", "are", "will", "do", "may", "might", "shall", "should",                  "must", "ought", "couldn't", "shouldn't", "wouldn't", "hasn't", "weren't"]

YES_NO_STEMS = ["may", "be", "would", "should", "do", "ought", "will", "must", "might", "have", "shall", "could", "can"]
CHOICE_WORDS = ["or", "either"]
QUESTION_WORDS = ["who", "what", "where", "when", "why", "how", "whose", "which", "whom"]

FIRST_PRONS = ["i", "me", "my", "mine", "we", "us", "our", "ours", "myself", "ourselves"]
SECOND_PRONS = ["you", "your", "yours", "yourself", "yourselves"]

ENTITY_MAP = defaultdict(lambda: [], 
              {"PERSON": ["PERSON", "ORG"],  # NER sometimes mistakes persons for orgs and vice versa
              "LOCATION": ["FAC", "GPE", "LOC", "NORP", "ORG"], # TODO: is ORG really necessary here?
              "DATE": ["DATE", "TIME", "CARDINAL"], # Sometimes labels [year] BC as CARDINAL ORG
              "NUMBER": ["PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"]})





# ============= simple helper functions =============

def get_text(path):
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


def get_questions(path):
  with open(path, "r") as file:
    return [line.replace("\n", "").strip() for line in file.readlines()]    

  
def get_test_cases(path):
  with open(path, "r") as file:
    return [line.replace("\n", "").strip().split("[SEP]") for line in file.readlines()]  


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


def post_process(sent):
  return sent.text.replace("\n", "").strip()





# ============= more complex helper functions =============

def get_question_ents(question):
  question_text = question.text
  ents_from_text = [((ent.text, ent.label_), ent) for ent in doc.ents if ent.text.lower() in question_text.lower()]
  ents_from_question = [((ent.text, ent.label_), ent) for ent in question.ents]
  return list(dict(ents_from_text + ents_from_question).values())


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
  

def filter_sents_ner(qword, qents, labels):
  
  def has_label(sent, exclude_ents_text):
    sent_labels = [ent.label_ for ent in sent.ents if ent.text not in exclude_ents_text]
    for label in labels:
      if label in sent_labels:
        return True
    return False

  # backup in case an entity was mislabled
  def has_query_ent(sent):
    for ent in qents:
      if ent.text in [sent_ent.text for sent_ent in sent.ents]: 
        return True
    return False
  
  if len(labels) > 0:
    exclude_ents_text = []
    if qword == "where":
      exclude_ents_text = list(set([ent.text for ent in qents if ent.label_ in ENTITY_MAP["LOCATION"]]))
      #print("exclude_ents_text: {}".format(exclude_ents_text))
    return [sent for sent in doc.sents if has_label(sent, exclude_ents_text)]
  else:
    return [sent for sent in doc.sents if len(stem_and_lower(sent)) > 0]


def process_question(question_text):
  question = nlp(question_text)
  question_word = ""
  question_start = -1
  last_pron_idx = -1

  has_yes_no = False
  answer_type = "NOCATEGORY"
  
  for (idx, tok) in enumerate([tok for tok in question]):
    if tok.text.lower() in QUESTION_WORDS:
      question_word = tok.text.lower()
      #question_start = idx
      break
    elif tok.lemma_.lower() in YES_NO_STEMS:
      has_yes_no = True
    elif tok.text.lower() in CHOICE_WORDS:
      answer_type = "CHOICE"
    
    if tok.text.lower() in (FIRST_PRONS + SECOND_PRONS):
      last_pron_idx = idx

      
  if last_pron_idx >= 0 and last_pron_idx < len(question) - 1:
    question = question[last_pron_idx + 1:]

  query_toks = process(question) # remove punctuation to get query tokens
  query_word = query_toks[0].text.lower()
  
  if len(question_word) > 0:
    if query_word == question_word:
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
  return (answer_type, query, question)


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


def get_bm25(query, sents_filtered, n):
  queries_stem = [stem_query(query, sent) for sent in sents_filtered]
  sents_stem_lower = [stem_and_lower(sent) for sent in sents_filtered]
  bm25 = BM25Okapi(sents_stem_lower)
  #assert(len(queries_stem) == len(sents_filtered))
  scored_sents = [(bm25.get_scores(queries_stem[i])[i], sents_filtered[i]) for i in range(len(sents_filtered))]
  #assert(len(scored_sents) == len(sents_filtered))
  #ave_dist_scores = [ave_dist_sent(query, sent_stem_lower) for sent_stem_lower in sents_stem_lower]
  #lccs_scores = [lccs(query_stem_lower, sent_stem_lower) for sent_stem_lower in sents_stem_lower]
  #print(ave_dist_scores)
  #scored_sents = list(zip(bm25.get_scores(query_stem_lower), ave_dist_scores, lccs_scores, sents_filtered))
  scored_sents.sort(key=lambda x: x[0], reverse=True)
  #scored_sents = [(round(entry[0], 1), round(entry[1], 1), entry[2], entry[3]) for entry in scored_sents]
  scored_sents = [(round(entry[0], 2), entry[1]) for entry in scored_sents]
  return scored_sents[:n]





# TODO: add comments
def pattern_match(question, question_ents, sents_filtered):
  
  question_processed = process(question)
  question_stem_lower = stem_and_lower(question)
  qword = question_stem_lower[0]
  
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
    
    
  def match_defn_nsubjpass(sent, subj, subj_phrase_lower=None):
    subj_word = subj.text.lower()
    for tok in sent:
      if tok.text.lower() == subj_word and (tok.pos_ in ["NOUN", "PROPN"]) and (tok.dep_ == "nsubjpass"):
        if subj_phrase_lower == None:
          return True
        else:
          # Handle multi-word subjects (subj_phrase_lower is not None)
          subj_phrase_lower_in_sent = " ".join(list([tok2.text.lower() for tok2 in tok.lefts]) + [tok.text.lower()])
          return subj_phrase_lower in subj_phrase_lower_in_sent
    return False
  
  
    
  def match_defn_appos(sent, subj, subj_phrase_lower=None):
    subj_word = subj.text.lower()
    for tok in sent:
      if tok.text.lower() == subj_word and (tok.pos_ in ["NOUN", "PROPN"])         and (tok.dep_ == "appos" or "appos" in [child.dep_ for child in tok.children]):
        if subj_phrase_lower == None:
          return True
        else:
          # Handle multi-word subjects (subj_phrase_lower is not None)
          subj_phrase_lower_in_sent = " ".join(list([tok2.text.lower() for tok2 in tok.lefts]) + [tok.text.lower()])
          return subj_phrase_lower in subj_phrase_lower_in_sent
    return False
  
  
  
  def get_matches(qword, subj, subj_phrase_lower=None):
    matches1 = [sent for sent in sents_filtered if match_defn12(sent, subj, 1, subj_phrase_lower)]
    matches2 = [sent for sent in sents_filtered if match_defn12(sent, subj, 2, subj_phrase_lower)]
    matches_appos = [sent for sent in sents_filtered if match_defn_appos(sent, subj, subj_phrase_lower)]
    matches_nsubjpass = [sent for sent in sents_filtered if match_defn_nsubjpass(sent, subj, subj_phrase_lower)]
    return [matches1, matches2, matches_nsubjpass, matches_appos]
  

  if qword in ["who", "what", "where", "when"]:
    #  and question_stem_lower[1] == "be"
    question_verbs = [tok for tok in question_processed if tok.pos_ in ["VERB", "AUX"]]
    """ question should only have one verb; we want to pattern match "What is a dog", but not
    "What is a dog classified as" (the ranking function can handle the second example) 
    """
    
#     if verbose:
#       print("Question verbs: {}".format(question_verbs))
      
    # Special case where we allow "where" questions to have a second verb: "located"
    # (for example, "Where is New York located?")
    where_located_case = (question_stem_lower[0] == "where" and len(question_verbs) == 2 and question_verbs[1].lemma_.lower() == "locate")
    if (len(question_verbs) == 1 and question_verbs[0].lemma_.lower() == "be") or where_located_case:
      verb_idx = 0
      if where_located_case:
        verb_idx = 1
      
#       if verbose:
#         print("Subj options: {}".format([tok for tok in list(question_verbs[verb_idx].children)]))
        
      subjs = [tok for tok in list(question_verbs[verb_idx].children) if (tok.dep_ in ["attr", "nsubj"])         and (tok.pos_ in ["NOUN", "PROPN"])]
      if verbose:
        #print("subj options: {}".format(list(question_verbs[verb_idx].children)))
        print("subjs: " + str(subjs))
      if len(subjs) == 1:
        subj = subjs[0]
        # Make sure the subject is an entity, not a common noun
        # (although should try to handle this case as well)
        #if len(nlp("{} is {}?".format(question_processed[0].text, subj)).ents) >= 1:
        if len(question_ents) >= 1:
          if qword == "who":
            if len(get_question_ents(nlp("Who is {}".format(subj)))) >= 1:
              # "who" question, can just use the single-word subj (likely a name)
              # TODO: what if subj is a last name, and multiple ents in the article share that last name?
              if verbose:
                print("Pattern match subj: " + subj.text)
              return get_matches(qword, subj)
          else:
            # not a "who" question, get the whole entity
            subj_phrase_lower = min([ent.text.lower() for ent in question_ents], key=lambda x: len(x))
            if verbose:
              print("subj_phrase_lower: {}".format(subj_phrase_lower))
            if subj.text.lower() in subj_phrase_lower:
              if subj.text.lower() == subj_phrase_lower:
                return get_matches(qword, subj)
              if verbose:
                print("Pattern match subj phrase: " + subj_phrase_lower)
              return get_matches(qword, subj, subj_phrase_lower)
              
  return [[], [], [], []]





def extract_answer(question_text, sentence_text):
  question_ids = bert_tokenizer.encode(question_text)
  sentence_ids = bert_tokenizer.encode(sentence_text)

  input_ids = bert_tokenizer.build_inputs_with_special_tokens(question_ids, sentence_ids)
  input_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)
#   if verbose:
#     print("Tokens:\n{}".format(input_tokens))
  token_type_ids = bert_tokenizer.create_token_type_ids_from_sequences(question_ids, sentence_ids)
  start_logits, end_logits = bert_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
  answer_score = float(torch.max(start_logits)) + float(torch.max(end_logits))  
  
  answer_preprocessed = " ".join(input_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1]).replace(" ##", "")
  sub1 = re.sub(r"([0-9]) (\.|,) ([0-9])", r"\1\2\3", answer_preprocessed)
  sub2 = re.sub(r"(.) ' (.{1,2})\b", r"\1'\2", sub1)
  sub3 = re.sub(r"(.) ' (.)", r"\1' \2", sub2)
  sub4 = re.sub(r"(.) , (.)", r"\1, \2", sub3)
  sub5 = re.sub(r" \)", r")", sub4)
  sub6 = re.sub(r"\( ", r"(", sub5)
  return (answer_score, sub6)





def get_answer_bert(question_text, sentence_text):
  question_ids = bert_tokenizer.encode(question_text)
  sentence_ids = bert_tokenizer.encode(sentence_text)

  input_ids = bert_tokenizer.build_inputs_with_special_tokens(question_ids, sentence_ids)
  input_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)

  token_type_ids = bert_tokenizer.create_token_type_ids_from_sequences(question_ids, sentence_ids)
  start_logits, end_logits = bert_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
  
  answer_preprocessed = " ".join(input_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1]).replace(" ##", "")
  sub1 = re.sub(r"(.) ' (.{1,2})\b", r"\1'\2", answer_preprocessed)
  sub2 = re.sub(r"(.) ' (.)", r"\1' \2", sub1)
  sub3 = re.sub(r"(.) , (.)", r"\1, \2", sub2)
  sub4 = re.sub(r" \)", r")", sub3)
  sub5 = re.sub(r"\( ", r"(", sub4)

  return (sub5, (float(torch.max(start_logits)) + float(torch.max(end_logits))))
  
  
def answer_question_bert(question_text):
  entries = [get_answer_bert(question_text, sent.text) for sent in doc.sents]
  #entries = [get_answer_bert(question_text, passage.replace("\n", "").strip()) for passage in passages]
  return max(entries, key=lambda x: x[1])[0]

  
def rank_sents_bert(question_text, n):
  entries = [(get_answer_bert(question_text, sent.text)[1], sent) for sent in doc.sents]
  entries.sort(key=lambda x: x[0], reverse=True)
  return [(round(x[0], 2), x[1]) for x in entries][:n]





def test_extract_answer(idx, sent=None):
  if sent == None:
    sent = get_best_sent(sample_questions[idx])
  return extract_answer(sample_questions[idx], sent)





def answer_yes_no(question_triple, question_ents):
  
  (answer_type, query, question) = question_triple
  subj = next((tok for tok in question if tok.dep_ == "nsubj"), None)
  
  if verbose:
    print("subj: {}".format(subj))

  if subj != None:
    subj_phrase_lower = subj.text.lower()
    subj_ent = min([ent for ent in question_ents if subj.text.lower() in ent.text.lower()], key=lambda ent: len(ent.text), default=None)
    
    if subj_ent != None and subj_ent.label_ != "PERSON":
      subj_phrase_lower = subj_ent.text.lower()
    
    if verbose:
      print("subj phrase lower: {}".format(subj_phrase_lower))
      
    ranked_sents = get_bm25(query, list(doc.sents), 5)
    threshold = ranked_sents[0][0] * 0.9
    best_sents = [entry for entry in ranked_sents if entry[0] >= threshold]
#     best_sent = post_process(ranked_sents[0][1])
#     best_sent_score = ranked_sents[0][0]
#     if verbose:
#       print("best sent: {}".format(best_sent))
#       print("best sent score: {}".format(best_sent_score))
      
    pred = question[subj.i+1:]
    match_words = stem_and_lower(pred, no_stop=True)
    
    if verbose:
      print("match_words: {}".format(match_words))
      
    for (score, sent) in best_sents:
      if subj_phrase_lower in sent.text.lower():
        sent_words = stem_and_lower(sent)
        if set(intersection(match_words, sent_words)) == set(match_words):
          return True
    
  return False





def get_best_sent(question_triple, question_ents):
  (answer_type, query, question) = question_triple
  question_stem_lower = stem_and_lower(question)
  #question_ents = get_question_ents(question)
  qword = question_stem_lower[0]
    
  sents_filtered = list(doc.sents)
  if qword == "where":
    sents_filtered = filter_sents_ner(qword, question_ents, ENTITY_MAP["LOCATION"])
  elif qword == "when":
    sents_filtered = filter_sents_ner(qword, question_ents, ENTITY_MAP["DATE"])
    
  matches = flatten(pattern_match(question, question_ents, sents_filtered))
  if len(matches) > 0:
    if verbose:
      print("Retrieval method: pattern match")
    return (True, [(-1, post_process(sent)) for sent in matches])
  else:
    if verbose:
      print("Retrieval method: ranking function")
    
    sents_filtered = filter_sents_ner(qword, question_ents, ENTITY_MAP[answer_type])
    if verbose:
      print("Sents remaining: {} of {}".format(len(sents_filtered), len(list(doc.sents))))
    if len(sents_filtered) == 0:
      sents_filtered = list(doc.sents)
      if verbose:
        print("Warning: all sentences filtered out, skipping NER filter step")
    
    ranked_sents = get_bm25(query, sents_filtered, 5)
    if len(ranked_sents) > 0:
        return (False, [(score, post_process(sent)) for (score, sent) in ranked_sents])
    else:
      return (False, [])



def answer_question(question_text):
  (answer_type, query, question) = process_question(question_text)
  question_ents = get_question_ents(question)
  
  if verbose:
    print("Question: {}\nQuery: {}\nAnswer type: {}".format(question.text, query, answer_type))
    
  if answer_type == "YESNO":
    ans_bool = answer_yes_no((answer_type, query, question), question_ents)
    if ans_bool:
      return "yes"
    else:
      return "no"
  else:
    (is_pattern_match, top_sent_entries) = get_best_sent((answer_type, query, question), question_ents)
    if len(top_sent_entries) == 0:
      return "unknown"
    
    if verbose:
        print("top_sent_entries: {}".format(top_sent_entries))
    
    if is_pattern_match:
      return max([extract_answer(question.text, sent_text) for (_, sent_text) in top_sent_entries], key=lambda x: x[0])[1]
    else:
      threshold = top_sent_entries[0][0] * 0.98
      best_sents_text = [sent_text for (score, sent_text) in top_sent_entries if score >= threshold]
      return max([extract_answer(question.text, sent_text) for sent_text in best_sents_text], key=lambda x: x[0])[1]


def run_tests(bert=False):
  test_cases = get_test_cases("tests/s{}a{}test.txt".format(set_num, doc_num))
  all_tests_passed = True
  for (question_text, answer_text) in test_cases:
    if bert:
      system_answer = answer_question_bert(question_text)
    else:
      system_answer = answer_question(question_text)
    if system_answer == answer_text:
      print("Test passed!")
    else:
      all_tests_passed = False
      print("Test failed\nQuestion: {}\nAnswer: {}\nYour answer: {}".format(question_text, answer_text, system_answer))
  if all_tests_passed:
    print("All tests passed!")
    

def update_tests():
  path = "tests/s{}a{}test.txt".format(set_num, doc_num)
  test_cases = get_test_cases(path)
  with open(path, "w+") as file:
    for (question_text, _) in test_cases:
      system_answer = answer_question(question_text)
      file.write(question_text + "[SEP]" + system_answer + "\n")
      
      
def create_tests():
  test_questions = get_questions("questions/s{}a{}.txt".format(set_num, doc_num))
  with open("tests/s{}a{}test.txt".format(set_num, doc_num), "w+") as file:
    for question_text in test_questions:
      system_answer = answer_question(question_text)
      file.write(question_text + "[SEP]" + system_answer + "\n")


if __name__ == '__main__':
  args = sys.argv
  if len(args) >= 3:
    set_num = int(args[1])
    doc_num = int(args[2])
    nlp = spacy.load("en_core_web_lg")
    text = get_text("data/Development_data/set{}/a{}.txt".format(set_num, doc_num))
    doc = nlp(text)
    passages = text.split("\n") 
    bert_tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    bert_model = BertForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")
    sample_questions = get_questions("questions/s{}a{}.txt".format(set_num, doc_num))
    verbose=False








