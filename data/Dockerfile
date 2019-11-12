# Ubuntu Linux as the base imag
FROM ubuntu:16.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8



#Install packages 
RUN apt-get -y update && \
    apt-get -y upgrade && \
	apt-get -y install python3-pip python3-dev && \
	pip3 install spacy numpy rank_bm25 nltk && \
	mkdir /QA
	#python3 -m spacy download en_core_web_lg && \


	

# Add the files
ADD ask /QA
ADD answer /QA

CMD ["chmod 777 ask"]
CMD ["chmod 777 answer"]

WORKDIR /QA

ENV PORT 9000

EXPOSE 9000
ENTRYPOINT [ "./start.sh" ]
