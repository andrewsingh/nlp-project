#!/bin/bash
export CLASSPATH=""; for file in `find . -name "*.jar"`; \
do export CLASSPATH="$CLASSPATH:`realpath $file`"; done \

nohup java -mx4g -cp "CoreNLP/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &
exec "$@"