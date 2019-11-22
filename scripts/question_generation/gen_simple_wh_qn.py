import spacy
import itertools
import numpy as np
import sys
from spacy import displacy
from spacy.symbols import ORTH
from collections import Counter
from collections import defaultdict

ENTITY_MAP = defaultdict(lambda: [], 
              {"PERSON": ["PERSON", "ORG"],  # NER sometimes mistakes persons for orgs and vice versa
              "LOCATION": ["FAC", "GPE", "LOC"], # TODO: is this enough? Need to do some more testing
              "DATE": ["DATE", "TIME", "CARDINAL"], # Sometimes labels [year] BC as CARDINAL ORG
              "NUMBER": ["PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"]})

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

#Matching based on dependency parses rules to generate patterns
def pattern_match(doc):
  def generate_who_qn(doc):
    sents_filtered = filter_sents_ner(doc, ENTITY_MAP["PERSON"])
    subj_dep = "attr"
    questions = []
    for sent in sents_filtered:
      for tok in sent:
        if tok.lemma_ == "be" and tok.pos_ in ["VERB", "AUX"]: 
          # is, are, am, be, etc.
          tok_ancestors = tok.ancestors
          for ancestor in tok.ancestors:
            if ancestor.ent_type_ in ENTITY_MAP["PERSON"]:
              #tok_children_text = [child.text.lower() for child in list(tok.children)]
              for child in tok.children:
                if child.dep_ == subj_dep:
                  questions.append("Who was " + ancestor.text + "?")
                  #TODO: Expand out children text
                  questions.append("Who was " + child.text +"?")
            elif ancestor.dep_ == subj_dep:
              for child in tok.children:
                if child.ent_type_ in ENTITY_MAP["PERSON"]:
                  questions.append("Who was " + ancestor.text + "?")
                  #TODO: Expand out children text
                  questions.append("Who was " + child.text +"?")
    return questions

  def generate_where_qn(doc):
    sents_filtered = filter_sents_ner(doc, ENTITY_MAP["LOCATION"])
    subj_dep = "attr"
    questions = []
    #Match for where patterns here.
    return questions

  #Currently just generating who qns.
  return generate_who_qn(doc)

def generate_qn(text):
    #Pattern match on 
    labels = ENTITY_MAP["PERSON"]
    sents_filtered = filter_sents_ner(doc, labels, query.ents)

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
    print(pattern_match(doc))
