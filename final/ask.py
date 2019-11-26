#!/usr/bin/env python3  

import warnings
warnings.filterwarnings("ignore")

import spacy
import itertools
import numpy as np
import sys
#import neuralcoref
from spacy import displacy
from spacy.symbols import ORTH
from collections import Counter
from collections import defaultdict
from did_final import generate_questions
from answer import *


ENTITY_MAP = defaultdict(lambda: [], 
              {"PERSON": ["PERSON", "ORG"],  # NER sometimes mistakes persons for orgs and vice versa
              "LOCATION": ["FAC", "GPE", "LOC"], # TODO: is this enough? Need to do some more testing
              "DATE": ["DATE", "TIME", "CARDINAL"], # Sometimes labels [year] BC as CARDINAL ORG
              "NUMBER": ["PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"]})

def get_text(path):
  def trim(lines):
    for i in range(len(lines)):
      line_txt = lines[i].replace("\n", "")
      if line_txt == "See also" or line_txt == "Notes" or line_txt == "References":
        return lines[:i]
    return lines

  with open(path, "r", encoding='utf-8') as file:
    #text = file.read()
    lines = file.readlines()
    sents = [line for line in trim(lines) if "." in line]

  return "".join(sents)

def stem_and_lower(sent, no_stop=False):
  return [tok.lemma_.lower() for tok in sent if not tok.is_punct and not (no_stop and tok.is_stop)]


def process(sent, no_stop=False):
  return [tok for tok in sent if not tok.is_punct and not (no_stop and tok.is_stop)]


def intersection(a, b):
  return [elem for elem in a if elem in b]


def flatten(nested_list):
  return [elem for sub_list in nested_list for elem in sub_list]

def serve_dep(ex):
  displacy.serve(ex, style="dep")


#Filter fn to grab sentences of various entities for pattern matching
def filter_sents_ner(doc, labels):
  def has_label(sent):
    sent_labels = [ent.label_ for ent in sent.ents]
    for label in labels:
      if label in sent_labels:
        return True
    return False

  if len(labels) > 0:	
    return [sent for sent in doc.sents if has_label(sent)]
  else:
    return [sent for sent in doc.sents if len(stem_and_lower(sent)) > 0]

# TODO: add comments
def pattern_match(question_text, question_ents, sents_filtered, verbose=False):
  
  question = nlp(question_text)
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
    
    
  def match_defn_appos(sent, subj, subj_phrase_lower=None):
    subj_word = subj.text.lower()
    for tok in sent:
      if tok.text.lower() == subj_word and (tok.pos_ in ["NOUN", "PROPN"]) and (tok.dep_ == "appos" or "appos" in [child.dep_ for child in tok.children]):
        if subj_phrase_lower == None:
          return True
        else:
          # Handle multi-word subjects (subj_phrase_lower is not None)
          subj_phrase_lower_in_sent = " ".join(list([tok2.text.lower() for tok2 in tok.lefts]) + [tok.text.lower()])
          return subj_phrase_lower in subj_phrase_lower_in_sent
    return False

  
  def get_matches(qword, subj, subj_phrase_lower=None, verbose=False):
    matches1 = [sent for sent in sents_filtered if match_defn12(sent, subj, 1, subj_phrase_lower)]
    matches2 = [sent for sent in sents_filtered if match_defn12(sent, subj, 2, subj_phrase_lower)]
    matches_appos = [sent for sent in sents_filtered if match_defn_appos(sent, subj, subj_phrase_lower)]
    return [matches1, matches2, matches_appos]
  

  if qword in ["who", "what", "where", "when"] and question_stem_lower[1] == "be":
    question_verbs = [tok for tok in question_processed if tok.pos_ in ["VERB", "AUX"]]
    """ question should only have one verb; we want to pattern match "What is a dog", but not
    "What is a dog classified as" (the ranking function can handle the second example) 
    """
    # Special case where we allow "where" questions to have a second verb: "located"
    # (for example, "Where is New York located?")
    where_located_case = (question_stem_lower[0] == "where" and len(question_verbs) == 2 and question_verbs[1].lemma_.lower() == "locate")
    #print("where_located_case: {}".format(where_located_case))
    if len(question_verbs) == 1 or where_located_case:
      verb_idx = 0
      if where_located_case:
        verb_idx = 1
      subjs = [tok for tok in list(question_verbs[verb_idx].children) if (tok.dep_ in ["attr", "nsubj"]) and (tok.pos_ in ["NOUN", "PROPN"])]
      if verbose:
        print("subjs: " + str(subjs))
      if len(subjs) == 1:
        subj = subjs[0]
        # Make sure the subject is an entity, not a common noun
        # (although should try to handle this case as well)
        #if len(nlp("{} is {}?".format(question_processed[0].text, subj)).ents) >= 1:
        if len(question_ents) >= 1:
          if qword == "who":
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
              
  return []

def generate_who_qn(doc):
  questions = []
  seen_ents = []
  for ent in doc.ents:
    if ent.label_ == "PERSON" and ent.text not in seen_ents:
      sub = ent[len(ent)-1].text
      sub_present =[i for i in seen_ents if sub in i]
      if len(sub_present) == 0: 
        seen_ents.append(ent.text)
        matched_patts = pattern_match("Who is " + ent.text+"?", [ent], doc.sents)
        if matched_patts != []:
          #print(matched_patts)
          if matched_patts[2] == []:
            questions.append("Who is " + ent.text +"?")
  return questions

def generate_what_qn(doc):
  questions = []
  seen_ents = []
  for ent in doc.ents:
    if ent.label_ in ["ORG"] and ent.text not in seen_ents:
      seen_ents.append(ent.text)
      matched_patts = pattern_match("What is " + ent.text+"?", [ent], doc.sents)
      if matched_patts != []:
        #print(matched_patts)
        if matched_patts[2] == []:
          left = ent[0]
          if left.text.lower() == "the":
            end = ent[len(ent)-1]
            if end.pos_ in ["NNS", "NNPS"]: 
              questions.append("What are " + ent.text +"?")
            else:
              questions.append("What is " + ent.text +"?")
          else:
            end = ent[len(ent)-1]
            if end.pos_ in ["NNS", "NNPS"]:
              questions.append("What are the " + ent.text +"?")
            else:
              questions.append("What is the " + ent.text +"?")
  return questions

'''def generate_where_qn(doc):
  questions = []
  seen_ents = []
  for ent in doc.ents:
    if ent.label_ in ["FAC", "LOC"] and ent.text not in seen_ents:
      seen_ents.append(ent.text)
      matched_patts = pattern_match("Where is " + ent.text+"?", [ent], doc.sents)
      if matched_patts != []:
        #print(matched_patts)
        if matched_patts[2] == []:
          left = ent[0]
          if left.text.lower() == "the":
            end = ent[len(ent)-1]
            if end.pos_ in ["NNS", "NNPS"]: 
              questions.append("Where are " + ent.text +"?")
            else:
              questions.append("Where is " + ent.text +"?")
          else:
            end = ent[len(ent)-1]
            if end.pos_ in ["NNS", "NNPS"]:
              questions.append("Where are the " + ent.text +"?")
            else:
              questions.append("Where is the " + ent.text +"?")
    if ent.label_ in ["GPE"] and ent.text not in seen_ents:
      seen_ents.append(ent.text)
      matched_patts = pattern_match("Where is " + ent.text+"?", [ent], doc.sents)
      if matched_patts != []:
        #print(matched_patts)
        if matched_patts[2] == []:
          left = ent[0]
          if left.text.lower() == "the":
            end = ent[len(ent)-1]
            if end.pos_ in ["NNS", "NNPS"]: 
              questions.append("Where are " + ent.text +"?")
            else:
              questions.append("Where is " + ent.text +"?")
          else:
            end = ent[len(ent)-1]
            if end.pos_ in ["NNS", "NNPS"]:
              questions.append("Where are " + ent.text +"?")
            else:
              questions.append("Where is " + ent.text +"?")
  return questions'''

def generate_simple_qns(doc):
  questions = []
  questions += generate_who_qn(doc)
  questions += generate_what_qn(doc)
  return questions

'''if __name__ == '__main__':
  args = sys.argv
  if len(args) >= 3:
    set_num = int(args[1])
    doc_num = int(args[2])
    text = get_text("data/Development_data/set{}/a{}.txt".format(set_num, doc_num))
    nlp = spacy.load("en_core_web_lg")
    #neuralcoref.add_to_pipe(nlp)
    sc1 = [{ORTH: "ca."}]
    nlp.tokenizer.add_special_case("ca.", sc1)
    doc = nlp(text)
    #doc._.has_coref
    #doc._.coref_clusters
    #print(doc._.coref_resolved)
    questions = generate_simple_qns(doc)
    questions +=
    for i in questions:
      print(i)
    sentence_list = generate_who_sents(doc)
    print(generate_who_questions(sentence_list))'''

def process_file(file):
  text = get_text(file)
  nlp = spacy.load("en_core_web_lg")
  sc1 = [{ORTH: "ca."}]
  nlp.tokenizer.add_special_case("ca.", sc1)
  doc = nlp(text)
  return doc

if __name__ == '__main__':
  args = sys.argv
  if len(args) >= 3:
    nlp = spacy.load("en_core_web_lg")
    doc_path = str(sys.argv[1])
    text = get_text(doc_path)
    doc = nlp(text) 
    num_qns = int(sys.argv[2])
    qn_output = generate_simple_qns(doc)
    qn_output += generate_questions(doc_path)
    if len(qn_output) > num_qns:
      qn_output = qn_output[:num_qns]

    for i in qn_output:
      print(i)
      print(answer_question(i, doc))

    '''nlp = spacy.load("en_core_web_lg")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    bert_model = BertForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")'''