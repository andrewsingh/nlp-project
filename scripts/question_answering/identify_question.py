import sys
import numpy
import collections
import nltk
import nltk.data
from rank_bm25 import BM25Okapi
import string
from sklearn.feature_extraction.text import TfidfVectorizer



yesnowords = ["can", "could", "would", "is", "does", "has", "was", "were", "had", "have", "did", "are", "will"]
commonwords = ["the", "a", "an", "is", "are", "were", "."]
questionwords = ["who", "what", "where", "when", "why", "how", "whose", "which", "whom"]
choicewords = ["or", "either", ]

def getquestiontype(qwords):
    qwords = nltk.word_tokenize(question.replace('?', ''))
    questionword = ""
    qidx = -1

    for (idx, word) in enumerate(qwords):
        if word.lower() in questionwords:
            questionword = word.lower()
            qidx = idx
            break
        elif word.lower() in yesnowords:
            return ("BINARY", qwords)
        elif word.lower() in choicewords:
            return ("CHOICE", qwords)

    if qidx < 0:
        return ("NOCATEGORY", qwords)

    if qidx > len(qwords) - 3:
        phrase = qwords[:qidx]
    else:
        phrase = qwords[qidx+1:]
    qtype = "NOCATEGORY"

    if questionword in ["who", "whose", "whom"]:
        qtype = "PERSON"
    elif questionword == "where":
        qtype = "LOCATION"
    elif questionword == "when":
        qtype = "TIME"
    elif questionword == "how":
        if phrase[0] in ["few", "little", "much", "many"]:
            qtype = "NUMBER"
            phrase = phrase[1:]
        elif phrase[0] in ["young", "old", "long"]:
            qtype = "TIME"
            phrase = phrase[1:]

    if questionword == "which":
        phrase = phrase[1:]
    if phrase[0] in yesnowords:
        phrase = phrase[1:]

    return (qtype, phrase)

if __name__ == '__main__' :
    question = "What genus is the domestic dog a member of"
    #question = "Which US state capital has the highest population?"
    print(question)
    (qtype, phrase) = getquestiontype(question)
    print(qtype, phrase)

    corpus = \
    "Hello there good man!. \
    It is quite windy in London. \
    How is the weather today?. \
    London is a beautiful city. \
    Domestic dog is a member of XYZ genus" 


    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    query = phrase[0].lower()
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)
    print(bm25.get_top_n(tokenized_query, corpus, n=1))

    corpus_tokens = nltk.sent_tokenize(corpus.lower())

    lemmer = nltk.stem.WordNetLemmatizer()
        
    def TfidfTokenizer(text):
        return (nltk.word_tokenize(question.lower().translate(dict((ord(punct), None) for punct in string.punctuation))))
    
    TfidfVec = TfidfVectorizer(tokenizer=TfidfTokenizer, stop_words='english')
    tfidf = TfidfVec.fit_transform(corpus_tokens)   
    print(TfidfVec.vocabulary_)
