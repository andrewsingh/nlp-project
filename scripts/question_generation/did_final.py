#!/usr/bin/env python
# coding: utf-8

import spacy
from pattern.en import conjugate, SINGULAR, INFINITIVE
from collections import defaultdict
from spacy.symbols import ORTH

nlp = spacy.load('en_core_web_lg')

ENTITY_MAP = defaultdict(lambda: [],
                         {"PERSON": ["PERSON"],
                          "LOCATION": ["FAC", "GPE", "LOC"],
                          "DATE": ["DATE", "TIME", "CARDINAL"],
                          "NUMBER": ["PERCENT", "MONEY", "QUANTITY",
                                     "CARDINAL", "ORDINAL"]})


def get_text(path):
    def trim(lines):
        for i in range(len(lines)):
            line_txt = lines[i].replace("\n", "")
            if line_txt == "See also" or line_txt == "Notes" \
                    or line_txt == "References":
                return lines[:i]
        return lines

    with open(path, "r", encoding='utf-8') as file:
        lines = file.readlines()
        sents = [line for line in trim(lines) if "." in line]

    return "".join(sents)


def has_label(sent, labels):
    sent_labels = [ent.label_ for ent in sent.ents]
    for label in labels:
        if label in sent_labels:
            return True
    return False


def stem_and_lower(sent, no_stop=False):
    return [tok.lemma_.lower() for tok in sent
            if not tok.is_punct and not
            (no_stop and tok.is_stop)]


def filter_sents_ner(doc, labels):
    if len(labels) > 0:
        return [sent for sent in doc.sents]
    else:
        return [sent for sent in doc.sents if len(stem_and_lower(sent)) > 0]


def generate_sents(doc):
    sents_filtered = filter_sents_ner(doc, ENTITY_MAP["PERSON"])
    return(sents_filtered)


# check_trans_verb() adapted from https://github.com/Mirith/Verb-categorizer/blob/master/transitivity.py
def check_trans_verb(token):
    indirect_object = False
    direct_object = False

    for tok in token.children:
        if(tok.dep_ == "iobj" or tok.dep_ == "pobj"):
            indirect_object = True
        if (tok.dep_ == "dobj" or tok.dep_ == "dative"):
            direct_object = True
    if direct_object and not indirect_object:
        return True
    else:
        return False


def modify_verb(verb):
    mod_verb = conjugate(verb,
                         tense=INFINITIVE,
                         parse=True,
                         number=SINGULAR,
                         person=3)
    return(mod_verb)


def extract_first_name(children):
    firstname = ''
    for left in children.lefts:
        if left.dep_ == "compound" or left.dep_ == 'amod':

            firstname += left.text + ' '
    return(firstname)


def additional_info(children, x):
    c = ''
    for right in children.children:
        if (right.dep_ == "pobj" and
            right.pos_ in ['NOUN', 'NUM']) \
                or right.dep_ == 'amod':
            c = right.text
            for left in right.lefts:
                x += left.text + ' '
    x += c
    return(x)


def phrase_question(ques_pattern):
    if len(ques_pattern['additional']) > 0:
        form = ques_pattern['person name'][0] + ' ' + \
            ques_pattern['modified verb'][0] + ' ' + \
            ques_pattern['additional'][0] + ' ?'
    else:
        form = ques_pattern['person name'][0] + ' ' + \
            ques_pattern['modified verb'][0] + ' ?'

    if len(ques_pattern['object']) > 0:
        form = ques_pattern['wh-type'][0] + ' ' + \
            ques_pattern['object'][0] + ' did ' + form
    else:
        form = ques_pattern['wh-type'][0] + ' ' + 'did ' + form
    return(form)


def questions(sentence):

    ques = []

    for token in sentence:
        TRANVERB = False

        if token.pos_ == 'VERB':
            TRANVERB = check_trans_verb(token)
        if TRANVERB:
            ques_pattern = {'modified verb': [],
                            'person name': [],
                            'additional': [],
                            'object': [],
                            'wh-type': ['What']
                            }
            flag = 0
            for children in token.children:

                if children.pos_ != "PRON" and \
                        children.dep_ == 'nsubj' and \
                        children.pos_ in ['PROPN']:
                    flag = 1

                    modified_verb = modify_verb(token.text)
                    ques_pattern['modified verb'].append(
                        modified_verb)

                    firstname = extract_first_name(children)
                    ques_pattern['person name'].append(
                        firstname + children.text)

                if children.dep_ == 'prt' and \
                        len(ques_pattern['modified verb']) > 0:
                    ques_pattern['modified verb'][-1] += ' ' + children.text

                if children.dep_ == 'dobj' and children.pos_ == 'NOUN':
                    if children.ent_type_ in ENTITY_MAP["PERSON"]:
                        ques_pattern['wh-type'] = 'Whom'

                    for ch in children.lefts:
                        if ch.dep_ == 'amod':
                            ques_pattern['object'].append(children.text)
                            break

                if children.dep_ == 'prep':
                    initial = children.text + ' '
                    final = additional_info(children, initial)
                    if len(final) > len(initial):
                        ques_pattern['additional'].append(final)

        if TRANVERB and flag:
            ques.append(phrase_question(ques_pattern))
    if len(ques) > 0:
        return(ques)


def generate_questions_per_line(sentence_list):
    questions_list = []
    for sentence in sentence_list:
        q = questions(sentence)
        if q is not None:
            questions_list.extend(q)
    return(questions_list)


def generate_questions(path):
    text = get_text(path)
    sc1 = [{ORTH: "ca."}]
    nlp.tokenizer.add_special_case("ca.", sc1)
    doc = nlp(text)
    sentence_list = generate_sents(doc)
    return(generate_questions_per_line(sentence_list))
