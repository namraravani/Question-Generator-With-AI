import flask
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import nltk as nlp
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import itertools
import logging
from typing import Optional, Dict, Union

from nltk import sent_tokenize

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import string
import math

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
logger = logging.getLogger(__name__)

app = Flask(__name__)
cors = CORS(app, origins='*')

class SubjectiveTest:
    def __init__(self, data, noOfQues):
        self.question_pattern = [
            "Explain in detail ",
            "Define ",
            "Write a short note on ",
            "What do you mean by "
        ]
        self.grammar = r"""
            CHUNK: {<NN>+<IN|DT>*<NN>+}
            {<NN>+<IN|DT>*<NNP>+}
            {<NNP>+<NNS>*}
        """
        self.summary = data
        self.noOfQues = noOfQues

    def generate_test(self):
        sentences = nlp.sent_tokenize(self.summary)
        cp = nlp.RegexpParser(self.grammar)
        question_answer_dict = dict()

        for sentence in sentences:
            tagged_words = nlp.pos_tag(nlp.word_tokenize(sentence))
            tree = cp.parse(tagged_words)
            for subtree in tree.subtrees():
                if subtree.label() == "CHUNK":
                    temp = ""
                    for sub in subtree:
                        temp += sub[0]
                        temp += " "
                    temp = temp.strip()
                    temp = temp.title()  # Capitalize only the first letter of each word
                    if temp not in question_answer_dict:
                        if len(nlp.word_tokenize(sentence)) > 20:
                            question_answer_dict[temp] = sentence
                    else:
                        question_answer_dict[temp] += sentence

        keyword_list = list(question_answer_dict.keys())
        question_answer = list()

        # Create a random number generator
        rng = np.random.default_rng()

        for _ in range(int(self.noOfQues)):
            rand_num = rng.integers(0, len(keyword_list))
            selected_key = keyword_list[rand_num]
            answer = question_answer_dict[selected_key]
            rand_num %= 4
            question = self.question_pattern[rand_num] + selected_key + "."
            question_answer.append({"Question": question, "Answer": answer})

        # Ensure that all questions are unique
        que = [q["Question"] for q in question_answer]
        ans = [q["Answer"] for q in question_answer]
        return que, ans


class E2EQGPipeline:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        use_cuda: bool
    ) :

        self.model = model
        self.tokenizer = tokenizer

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]

        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"

        self.default_generate_kwargs = {
            "max_length": 256,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

    def __call__(self, context: str, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)

        # TODO: when overrding default_generate_kwargs all other arguments need to be passsed
        # find a better way to do this
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs

        input_length = inputs["input_ids"].shape[-1]

        # max_length = generate_kwargs.get("max_length", 256)
        # if input_length < max_length:
        #     logger.warning(
        #         "Your max_length is set to {}, but you input_length is only {}. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=50)".format(
        #             max_length, input_length
        #         )
        #     )

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        questions = prediction.split("<sep>")
        questions = [question.strip() for question in questions[:-1]]
        return questions

    def _prepare_inputs_for_e2e_qg(self, context):
        source_text = f"generate questions: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"

        inputs = self._tokenize([source_text], padding=False)
        return inputs

    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs

# Now you can instantiate the E2EQGPipeline class directly
model_name = "valhalla/t5-small-e2e-qg"  # Specify the model name or path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
use_cuda = True  # Set to True if you want to use GPU

qg_pipeline = E2EQGPipeline(model=model, tokenizer=tokenizer, use_cuda=use_cuda)


def cleaning_and_calculating_frequencies(paragraph):
    # Tokenize the paragraph into words
    words = word_tokenize(paragraph)

    # Remove punctuation and special characters
    words = [word for word in words if word.isalnum()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    word_freq = {}
    total_words = 0

    # Count the frequency of each word and calculate the total number of words
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
        total_words += 1

    # Calculate the normalized frequency of each word
    normalized_freq = {word: freq / total_words for word, freq in word_freq.items()}

    return normalized_freq


def compute_joint_probability(question_probs, paragraph_probs):
    """
    Compute the joint probability distribution of word pairs between the question and the paragraph.

    Args:
        question_probs (dict): Probability distribution of individual words in the question.
        paragraph_probs (dict): Probability distribution of individual words in the paragraph.

    Returns:
        dict: A dictionary representing the joint probability distribution of word pairs.
    """
    joint_probs = {}

    # Compute the joint probability for each word pair
    for word_q, prob_q in question_probs.items():
        for word_p, prob_p in paragraph_probs.items():
            joint_probs[(word_q, word_p)] = prob_q * prob_p

    # Optional: Normalize the joint probabilities
    total_joint_prob = sum(joint_probs.values())
    joint_probs = {pair: prob / total_joint_prob for pair, prob in joint_probs.items()}

    

    return joint_probs



def compute_mutual_information(joint_probs, question_probs, paragraph_probs):

    """
    Compute the mutual information between the question and the paragraph using the joint probability distribution.

    Args:
        joint_probs (dict): Joint probability distribution of word pairs between the question and the paragraph.
        question_probs (dict): Probability distribution of individual words in the question.
        paragraph_probs (dict): Probability distribution of individual words in the paragraph.

    Returns:
        float: Mutual information between the question and the paragraph, normalized to [0, 1].
    """
    mutual_info = 0.0

    

    # Compute mutual information using the joint probability distribution
    for word_q, prob_q in question_probs.items():
        for word_p, prob_p in paragraph_probs.items():
            joint_prob = joint_probs.get((word_q, word_p), 0.0)

            if joint_prob > 0:
                
                mutual_info += joint_prob * math.log(joint_prob / (prob_q * prob_p))

                
    # Normalize mutual information to [0, 1]
    max_possible_mi = -sum(prob * math.log(prob) for prob in paragraph_probs.values())
    min_possible_mi = 0
    normalized_mi = 1 - ((mutual_info - min_possible_mi) / (max_possible_mi - min_possible_mi))

    
   

    return normalized_mi



def calculate_the_mutual_information(paragraph, questions):
    cleaned_paragraph = cleaning_and_calculating_frequencies(paragraph)
    question_mutual_info = {}

    for question in questions:
        cleaned_question = cleaning_and_calculating_frequencies(question)
       
        joint_probability = compute_joint_probability(cleaned_question, cleaned_paragraph)
        
        mutual_information = compute_mutual_information(joint_probability, cleaned_question, cleaned_paragraph)
        question_mutual_info[question] = mutual_information

    return question_mutual_info

def categorize_mutual_information(scores):
    min_score = min(scores.values())
    max_score = max(scores.values())

    # Calculate the range between the lowest and highest scores
    score_range = max_score - min_score

    # Define thresholds for categorization
    low_threshold = min_score + (score_range * 0.33)
    high_threshold = min_score + (score_range * 0.67)

    categorized_scores = {}
    for question, score in scores.items():
        if score >= high_threshold:
            categorized_scores[question] = "HARD"
        elif score >= low_threshold:
            categorized_scores[question] = "MEDIUM"
        else:
            categorized_scores[question] = "EASY"
    return categorized_scores

@app.route("/api/calculate-mutual-info", methods=['POST'])
def calculate_mutual_info():
    # Get the input text from the request
    input_data = request.json
    text = input_data['text']

    # Generate questions
    number_of_questions = 200  # Set the number of questions you want to generate
    test_generator = SubjectiveTest(text, number_of_questions)
    questions, _ = test_generator.generate_test()

    Accurate_questions = qg_pipeline(text)
    questions, answers = test_generator.generate_test()

    question_list = []

    for i, (question_in, answer) in enumerate(zip(questions, answers), 1):
        question_list.append(question_in)

    # Combine questions from both sources
    combined_question_list = list(set(Accurate_questions)) + list(set(question_list) - set(Accurate_questions))

    # Calculate the mutual information
    output = calculate_the_mutual_information(text, combined_question_list)

    # Categorize the mutual information
    categorized_output = categorize_mutual_information(output)

    return jsonify(categorized_output)

if __name__ == "__main__":
    app.run(debug=True,port=8080)
