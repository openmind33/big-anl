#!/usr/bin/env /c/Apps/Anaconda3/python

"""
# 
# [quick learn, word2vec, Unsupervised]
import codecs
import gensim, logging
from gensim.models import word2vec, Word2Vec
from pprint import pprint
sentences = word2vec.Text8Corpus('c:/Home/OneDrive/e200Prg/e200TDA/text8')
model = word2vec.Word2Vec(sentences, size=200, workers=12, min_count=3, sg=0, window=8, iter=15, sample=1e-4, negative=25)
word_similarity = model.similarity('king','queen')
pprint(word_similarity)
word_matching = model.most_similar(positive=['king','queen'],negative=['man'],topn=100)
for i in range(len(word_matching)):
	print(i, word_matching[i])
"""
print(__doc__)
import codecs
import gensim, logging
from gensim.models import word2vec, Word2Vec
from pprint import pprint


from itertools import chain
from glob import glob

inputText = 'C:/Home/data/file/machine_learning.txt'
lines = set(chain.from_iterable(codecs.open(f, 'r',encoding="utf-8") for f in glob(inputText)))
lines = [line.lower() for line in lines]
with codecs.open(inputText, 'w',encoding="utf-8") as out:
    out.writelines(sorted(lines))
sentences = word2vec.Text8Corpus(inputText)
model = word2vec.Word2Vec(sentences, size=200, workers=12, min_count=3, sg=0, window=8, iter=15, sample=1e-4, negative=25)

word1 = 'machine'
word2 = 'learning'
word_similarity = model.similarity(word1,word2)
print(word1,word2)
pprint(word_similarity)

word_matching = model.most_similar(positive=[word1,word2],negative=['computer'],topn=1000)
print(word1,word2)
for i in range(len(word_matching)):
	print(i, word_matching[i])

# sentencesCorpus = word2vec.Text8Corpus('data/sentences.txt')
# sentencesCorpus = word2vec.Text8Corpus('data/shakespeare.txt')
'''
word_similarity = model.similarity('machine','computer')
pprint(word_similarity)
word_matching = model.most_similar(positive=['machine','computer'],negative=['learning'],topn=100)
for i in range(len(word_matching)):
	print(i, word_matching[i])



model = word2vec.Word2Vec(sentencesCorpus, size=200, workers=12, min_count=1, sg=0, window=8, iter=15, sample=1e-4, negative=25)
print(model)
# model.save('sentencesModel')
# fname = 'sentencesModel'
# model = Word2Vec.load(fname)
# print(model)
# pprint(model.vocab)
print('max =', len(model.vocab) -1)
wordVocab = [k for (k, v) in model.vocab.iteritems()]
print(wordVocab)
result = model.most_similar(positive=['innovation', 'market'], negative=['new'])
pprint(result)
result = model.doesnt_match("innovation")
print(result)
result = model.similarity('innovation','new')
print(result)
# vectors in word-by-word
print(model['innovation'])
'''


'''
# print(model)
# model.accuracy("sentences_test.txt")
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('text8')

# train the skip-gram model; default window=5, some hours
model = word2vec.Word2Vec(sentences, size=200)
model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10)
# pickle the entire model to disk, so we can load&resume training later
model.save('text8.model')
# store the learned weights, in a format the original C tool understands
model = word2vec.Word2Vec.load_word2vec_format('vectors.bin', binary=True)
model.most_similar(['girl', 'father'], ['boy'], topn=3)
more_examples = ["he his she", "big bigger bad", "going went being"]
for example in more_examples:
	a, b, x = example.split()
	predicted = model.most_similar([x, b], [a])[0][0]
	print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))
# which word doesn't go with the others?
model.doesnt_match("breakfast cereal dinner lunch".split())
'''
'''
import gensim.models
import time
time1 = time.time()

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
modelbase = gensim.models.Word2Vec()
sentences2 = gensim.models.word2vec.Sentences("text8-queen")
modelbase.build_vocab(sentences2)
modelbase.train(sentences2)
modelbase.save_word2vec_format("wordvectors/model-text8-queen-only")
modelbase.accuracy("questions-words.txt")

model = gensim.models.Word2Vec()
sentences = gensim.models.word2vec.Sentences("text8-rest")
model.build_vocab(sentences)
model.train(sentences)
model.save_word2vec_format("model-text8-rest")
model.accuracy("questions-words.txt")

sentences2 = gensim.models.word2vec.Sentences("text8-queen")
model.update_vocab(sentences2)
model.train(sentences2)
model.save_word2vec_format("wordvectors/model-text8-queen")
model.accuracy("questions-words.txt")

model1 = gensim.models.Word2Vec()
sentences = gensim.models.word2vec.Sentences("text8-all")
model1.build_vocab(sentences)
model1.train(sentences)
model1.save_word2vec_format("wordvectors/model-text8-all")
model1.accuracy("questions-words.txt")
print ("total time: %s" % (time.time() - time1))
'''
