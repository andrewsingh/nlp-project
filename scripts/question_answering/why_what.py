#!/usr/bin/env python
# coding: utf-8

# In[145]:


import spacy
from spacy import displacy
# import pattern.en as en


# In[ ]:


nlp = spacy.load('en_core_web_lg')


# In[ ]:


def get_text(path):
    def trim(lines):
        for i in range(len(lines)):
            line_txt = lines[i].replace("\n", "")
            if line_txt == "See also" or line_txt == "Notes" or line_txt == "References":
                return lines[:i]
        return lines

    with open(path, "r", encoding='utf-8') as file:
        lines = file.readlines()
        sents = [line for line in trim(lines) if "." in line]
        doc = nlp("".join(sents))
        sentence = [sent.text for sent in doc.sents]
    return sentence


# In[155]:


sentences = get_text('path to file')


for sentence in sentences:
    doc = nlp(sentence)
    ques = ['Wh-q']

    verb = ['VB',
            'VBD',
            'VBG',
            'VBN',
            'VBP',
            'VBZ',
            ]
    noun = ['NNP', 'NNPS']
    prepo = ''

    for token in doc:
        if token.dep_ == 'nsubj' and token.tag_ in noun and token.head.lemma_ == 'be':
            flag = 1
            inter = []
            subj = token.text

            cnt = 0
            for t in token.head.rights:
                if t.dep_ == 'acomp':
                    flag = 0

            if flag == 0:
                break

            for t in token.lefts:
                if t.dep_ == 'det':
                    inter.append(t.text)
                if t.dep_ == 'compound':
                    inter.append(t.text)

            if len(inter) > 1:
                ques.extend([token.head.text, inter[0], inter[1], token.text])
            elif len(inter) == 1:
                ques.extend([token.head.text, inter[0], token.text])
            else:
                ques.extend([token.head.text, token.text])

        if token.dep_ == 'prep' and token.head.text == subj:
            prepo = token.text
            ques.append(token.text)

        if token.dep_ == 'pobj' and token.head.text == prepo:
            ques.append(token.text)
            break

    if len(ques) > 1:
        print(sentence)
        print(ques)
