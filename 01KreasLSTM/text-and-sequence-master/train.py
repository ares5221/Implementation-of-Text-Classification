import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt


# settings
max_len = 100
training_samples = 200
validation_samples = 10000
max_words = 10000
embedding_dim = 100


def process_data():
    '''
    处理IMDB数据,将数据按标签分为pos，neg
    :return: labels,texts
    '''
    imdb_dir = 'D:\\text2sequences\\aclImdb\\aclImdb'
    train_dir = os.path.join(imdb_dir, 'test')

    labels = []
    texts = []

    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname), 'r', encoding='UTF-8')
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)

    return labels, texts


def tokennize_test_data():
    labels, texts = process_data()
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts=texts)
    sequences = tokenizer.texts_to_sequences(texts=texts)
    x_test = pad_sequences(sequences, maxlen=max_len)
    y_test = np.asarray(labels)

    return x_test, y_test



def tokennize_data():
    '''
    将text向量化，切分训练集和验证集
    :return:x_train, y_train, x_val, y_val即训练集和验证集的label和text
    '''

    labels, texts = process_data()
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts=texts)
    sequences = tokenizer.texts_to_sequences(texts=texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=max_len)

    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]

    return x_train, y_train, x_val, y_val, word_index


def parse_word_embedding(word_index):
    '''
    将预计算的词向量空间的word建立索引和矩阵
    :return:
    '''
    glove_dir = 'D:\\text2sequences\\glove.6B'

    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), 'r', encoding='UTF-8')
    for  line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((max_words, embedding_dim))

    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def train_model():
    '''
    训练模型
    :return:训练时loss,acc
    '''
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_len))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # 将GLOVE加载到模型中
    x_train, y_train, x_val, y_val, word_index = tokennize_data()
    # x_test, y_test = tokennize_test_data()
    embedding_matrix = parse_word_embedding(word_index)
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_val, y_val))
    model.save('pre_trained_glove_model_1.h5')

    # 评估专用
    # model.load_weights('pre_trained_glove_model_1.h5')
    # test_acc = model.evaluate(x_test, y_test)
    # print(test_acc)



def plott_results():
    '''
    作图
    '''
    history = train_model()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_model()
