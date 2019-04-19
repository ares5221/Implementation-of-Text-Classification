import numpy as np
#
# samples = ['The cat sat on the mat.', 'The dog ate my homework.']
#
# # 10
# # 定义一个集合
# token_index = {}
# for sample in samples:
#     for word in sample.split():
#         if word not in token_index:
#             token_index[word] = len(token_index) + 1
#
#
# max_length = 10
# results = np.zeros(shape=(len(samples),
#                           max_length,
#                           max(token_index.values()) + 1))
#
# # print(results) 2, 10, 11
# for i, sample in enumerate(samples):
#     for j, word in list(enumerate(sample.split()))[:max_length]:
#         index = token_index.get(word)
#         results[i, j, index] = 1.
#
#
# print(results)

# import string
# samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# characters = string.printable
# token_index = dict(zip(range(1, len(characters) + 1), characters))
#
# max_length = 50
# results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))
# for i, sample in enumerate(samples):
#     for j, character in enumerate(sample):
#         for key, value in token_index.items():
#             if value == character:
#                 index = key
#                 results[i, j, index] = 1.
#
#         # index = token_index.get(character)
#
# print(results)
# from keras.preprocessing.text import Tokenizer
#
# samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# tokenizer = Tokenizer(num_words=1000)
# tokenizer.fit_on_texts(samples)
#
# sequences = tokenizer.texts_to_sequences(samples)
# one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
#
# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1


print(results)