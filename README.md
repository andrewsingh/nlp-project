# nlp-project
Project for 11-411 Natural Language Processing

## Running the question answering script
Checkout the repo and cd to the root.
Then install the requirements by running \
`pip install scripts/question_answering/qa_requirements.txt` 

Once you have the requirements installed, run the script interactively by entering \
`python3 -i scripts/question_answering/answer_question.py [set_num] [doc_num]` \
To test a question, enter \
`get_best_sent(<question_text>)`

### Example
To test questions about article 1 from set 1 (Old Kingdom), enter \
`python3 -i scripts/question_answering/answer_question.py 1 1` \
Then, to test the question "Who is King Djoser?", enter \
`get_best_sent("Who is King Djoser?")` 

Note: a number of sample questions are included for the very first article (set 1 article 1), and are stored in the array `S1A1`.


