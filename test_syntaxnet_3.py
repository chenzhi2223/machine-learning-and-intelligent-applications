# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:40:06 2016

@author: zhi_chen
In this small program, we try to utilize the parsed dependency
via Google's Syntaxnet to predict the party names of a document.

An updated version of bag_of_words with parsed dependency taken
into account is used to predict the party names.
"""

import subprocess
import os
import re
from collections import Counter

os.chdir('/Users/apple/models/syntaxnet')


"""
sample list where the elements cannot be party names.
More can be added in the future.
"""
non_party_names = ['term', 'terms', 'application', 'agreement', 'service',
                   'Definitions', 'price']


def read_text(filepath, filename):
    """
    returns the content of the input file.

    input:
    type filepath: string
    param filepath: path of the input file.

    type filename: string
    param filename: name (inclusive of file type) of the input file.
    """
    with open(filepath + filename, 'r') as f:
        text = f.read()
    return text


def split_sentence(text):
    """
    input:
    text: a string which is the content of document

    output:
    returns the splitted sentence using regular expressions.
    """
    return re.split(r' *[\.\?!][\'"\)\]]* * +', text)
    # return re.split(r'(?<=[^A-Z].[].?) +(?=[A-Z])', text)


def test_split(f):
    """
    test for split_sentence(text) function.

    input:
    f: a file object for writting test results.
    """

    splitted = split_sentence("The Organizer's payment method is used "
                              "to pay for any purchase initiated by a "
                              "Family member in excess of any store "
                              "credit in such initiating Family "
                              "member’s account. Products are "
                              "associated with the account of the "
                              "Family member who initiated the "
                              "transaction.")
    f.write('\n---------\n')
    f.write('test of sentence splitting:\n')
    for sentence in splitted:
        f.write(sentence + '\n\n')
    f.write('\n---------\n')


def parsed(feedin):
    """
    input:
    feedin: a string (sentence) for parsing.

    output:
    parsed: returns the parsed dependency of the input sentence \
    using syntaxnet.
    """
    try:
        parsed = subprocess.check_output([
            "echo '%s' | bazel-bin/syntaxnet/parser_eval \
             --input stdin \
             --output stdout-conll \
             --model syntaxnet/models/parsey_mcparseface/tagger-params \
             --task_context syntaxnet/models/parsey_mcparseface/context.pbtxt \
             --hidden_layer_sizes 64 \
             --arg_prefix brain_tagger \
             --graph_builder structured \
             --slim_model \
             --batch_size 1024 | bazel-bin/syntaxnet/parser_eval \
             --input stdin-conll \
             --output stdout-conll \
             --hidden_layer_sizes 512,512 \
             --arg_prefix brain_parser \
             --graph_builder structured \
             --task_context syntaxnet/models/parsey_mcparseface/context.pbtxt \
             --model_path syntaxnet/models/parsey_mcparseface/parser-params \
             --slim_model --batch_size 1024" % feedin,
        ], shell=True)
        # print parsed
        return parsed
    except subprocess.CalledProcessError as e:
        print 'error info:\n', e.output


def test_parsed(f):
    """
    test for parsed(feedin) function.

    input:
    f: a file object for writting test results.
    """

    output1 = parsed(
        'You can use these products if you did not agree on their terms!')
    output2 = parsed('Does Beagle love AI?')
    f.write('test of sentence parsed:\n')
    f.write('\n---------\n')
    f.write(output1 + '\n')
    f.write(output2 + '\n')
    f.write('\n---------\n')


def word_array(parsed_sentence):
    """
    input:
    parsed_sentence: output of the parsed result using syntaxnet.

    output:
    parsed_array: returns a two-dimension array for one sentence, \
    each word a row.

    others:
    In the parsed structure each word has 10 elements. For instance:
    the parsed structure for 'Does Beagle love AI?' is:
    1	Does	_	VERB	VBZ	_	3	aux	_	_
    2	Beagle	_	NOUN	NNP	_	3	nsubj	_	_
    3	love	_	NOUN	NN	_	0	ROOT	_	_
    4	AI	_	NOUN	NNP	_	3	dobj	_	_
    5	?	_	.	.	_	3	punct	_	_

    variable in use:
    parsed_list: a list of words of the input (parsed_sentence).
    wordcount: number of words in the original sentence \
    (where 10 is the number of items for each word in the parsed sentence).
    """
    parsed_list = parsed_sentence.split()
    wordcount = len(parsed_list) / 10
    parsed_array = [
        [parsed_list[j * 10: (j + 1) * 10]
         ]
        for j in range(wordcount)
    ]
    return parsed_array


def test_word_array(f):
    """
    test word_array(parsed_sentence) function.

    input:
    f: a file object for writting test results.
    """
    parsed_output = parsed('Beagle is a smart dog!')
    output = word_array(parsed_output)
    f.write('test of wrod array splitted:\n')
    f.write('\n---------\n')
    for index, item in enumerate(output):
        f.write(str(item))
        f.write('\n')
    f.write('\n---------\n')


def count_words(bag_of_words, parsed_array):
    """
    input:
    bag_of_words: a dictionary contains word as key and word count
    as the value.
    parsed_array: a two-dimension array from a parsed_sentence,
    with each word as a row, and each row contains 10 itmes.

    output:
    bag_of_words: returns the accumulated count of the number of \
    valid words for party name prediction in the dictionary.

    Others:
    wordcount: number of words in parsed_array.
    Important items in parsed_array of each word:
    item[1]: word i itself.
    item[3]: POS tag of word i.
    item[7]: parsed dependency.
    valid_tag: list of valid tags whose related word can be party names.
    valid_dependency: list of valid_dependency whose related word can be
    party names.

    Please see http://universaldependencies.org/u/dep/index.html
    for more details on dependents of clausal predicates
    Some core dependents are listed:
    'nsubj': nominal subject.
    'dobj': direct object
    'iobj': indirect object
    'csubj': clausal subject
    Note that all these are dependent on verbs which can be the root of
    the parsed sentence

    Logic:
    selecting a word of type 'noun' and 'pron' which can be party names, \
    and only these with parsed label in the label list valid_dependency is \
    selected, as the party name is probablily used as the subject or \
    objective of the sentence.
    Note that they may also be used as objective, but here we only consider \
    the nominal subject case.
    Further thought:
    some words are definitely not party names. It might be \
    helpful to build such a list of these words for better prediction. \
    However, it needs more efforts to maintain it and \
    is simply used as a toy model here.
    """
    valid_tag = ['NOUN', 'PRON']
    valid_dependency = ['nsubj', 'dobj', 'iobj', 'csubj']
    # wordcount = len(parsed_array)
    for index, item in enumerate(parsed_array):
        print item
        if (item[0][1] not in non_party_names and
            item[0][3] in valid_tag and
                item[0][7] in valid_dependency):
            # get uppercase of the word
            valid_word = (item[0][1]).upper()
            if valid_word not in bag_of_words.keys():
                bag_of_words[valid_word] = 1
            else:
                bag_of_words[valid_word] = bag_of_words[valid_word] + 1
    return bag_of_words


def test_count_words(f):
    """
    test count_words funciton.

    input:
    f: a file object for writting test results.
    """

    splitted = split_sentence(
        "He said he would pay you five "
        "pounds a week if he could have it on my own terms. "
        "He is a poor woman, sir, and Mr. Warren earns little, "
        "and the money meant much to me.  He took out a "
        "ten-pound note, and he held it out to me then and there."
    )
    bag_of_words = dict()
    for sentence in splitted:
        parsed_sentence = parsed(sentence)
        print parsed_sentence
        parsed_array = word_array(parsed_sentence)
        bag_of_words = count_words(bag_of_words, parsed_array)
    f.write('test of words count:\n')
    f.write('\n---------\n')
    for key in bag_of_words.keys():
        print key + ': ' + str(bag_of_words[key]) + '\n'
        f.write(key + ': ' + str(bag_of_words[key]) + '\n')
    f.write('\n---------\n')


if __name__ == "__main__":
    # get the content of the agreement file.
    text = read_text('/Users/apple/Documents/', 'itune_usa.txt')
    # testing of designed functions.
    with open('/Users/apple/Documents/testlog.txt', 'w') as f:
        test_split(f)
        test_parsed(f)
        test_word_array(f)
        test_count_words(f)
    splitted = split_sentence(text)
    sentence_num = len(splitted)
    # storing valid words and display the two \
    # words with the highest frequency as the party name.
    bag_of_words = dict()
    # set the counter to zero, as only sentences
    # at the beginning of the document will be considered.
    # note that this is just testing for efficiency, not for real use.
    counter = 0
    interval = min(50, sentence_num / 2)
    for sentence in splitted:
        # print sentence
        if ((sentence is not None and len(sentence) > 0) and
                (counter < interval or counter >= sentence_num - interval)):
            parsed_sentence = parsed(sentence)
            if parsed_sentence is not None and len(parsed_sentence) > 0:
                parsed_array = word_array(parsed_sentence)
                # count the number of noun words and store in dict.
                bag_of_words = count_words(bag_of_words, parsed_array)
        counter = counter + 1
    """
    get the two most frequent noun words as parties.
    this is not necessarily right as more insights and features should be
    considered for more accuracy result.
    """
    get_sorted = Counter(bag_of_words).most_common()
    party1 = get_sorted[0][0]
    party2 = get_sorted[1][0]
    # writing to the output file:
    with open('/Users/apple/Documents/output.txt', 'w') as f:
        f.write('Agreement Name, Party 1: %s, Party 2: %s'
                % (party1, party2))
