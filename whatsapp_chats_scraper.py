import gensim

import time
import datetime
import numpy as np
import pandas as pd


def train_model(data, epochs=100, vect_size=100):
    """
    Train a model using the provided data
    :param data: list of sentences
    :return: a trained model using default parameters
    """
    sentences_vec = []
    for sentence in data:
        sentences_vec.append(sentence.split())

    w2v_model = gensim.models.Word2Vec(min_count=1, size=vect_size)  # an empty model, no training
    w2v_model.build_vocab(sentences_vec)  # can be a non-repeatable, 1-pass generator
    w2v_model.train(sentences_vec, total_examples=w2v_model.corpus_count, epochs=epochs) #epochs=w2v_model.iter)
    return w2v_model


def save_model(model, path):
    """
    Save the given model into the given path
    :param model: a trained model
    :param path: the file path
    :return: nothing
    """
    # model.save(path)
    gensim.models.KeyedVectors.save_word2vec_format(model.wv, path, binary=True)


def load_model(path):
    """
    Loads a model from the given path
    :param path: the file containing the saved model
    :return: the model, if the file does not exist you will notice.
    """
    return gensim.models.Word2Vec.load(path)


def online_train(model, more_sentences):
    """
    Trainds a given model on new data.
    :param model: The new model trained on the new data on top of the data is was already trained on.
    :param more_sentences: the new sentences list for the online training.
    :return: the new trained model
    """
    sentences_vec = []
    for sentence in more_sentences:
        sentences_vec.append(sentence.split())
    model.build_vocab(sentences_vec, update=True)
    model.train(sentences_vec, total_examples=model.corpus_count, epochs=model.iter)

    return model


def get_embeddings(model):
    """
    gets the words embedding in a dictionary format.
    :param model: the trained model
    :return: the words embeddings in a dictionary format.
    Each key is a word and the value is the word associated vector.
    """
    embeddings = {}
    for word in model.wv.vocab:
        embeddings[str(word)] = model[word]
    return embeddings


def benchmark():
    # not edited: using 25 KB and 50 MB files only for generating o/p -> comment next line for using all 5 test files
    input_data_files = ['data/lee_background.cor']
    print(input_data_files)

    train_time_values = []
    seed_val = 42
    sg_values = [0, 1]
    hs_values = [0, 1]

    for data_file in input_data_files:
        print('Data file:', data_file)
        data = gensim.models.word2vec.LineSentence(data_file)
        for sg_val in sg_values:
            for hs_val in hs_values:
                for loss_flag in [True, False]:
                    time_taken_list = []
                    for i in range(3):
                        start_time = time.time()
                        w2v_model = gensim.models.Word2Vec(data, compute_loss=loss_flag, sg=sg_val, hs=hs_val, seed=seed_val)
                        time_taken_list.append(time.time() - start_time)

                    time_taken_list = np.array(time_taken_list)
                    time_mean = np.mean(time_taken_list)
                    time_std = np.std(time_taken_list)
                    train_time_values.append({'train_data': data_file, 'compute_loss': loss_flag, 'sg': sg_val, 'hs': hs_val, 'mean': time_mean, 'std': time_std})

    train_times_table = pd.DataFrame(train_time_values)
    train_times_table = train_times_table.sort_values(by=['train_data', 'sg', 'hs', 'compute_loss'], ascending=[False, False, True, False])
    print(train_times_table)


def clean_line(line, strict_arabic=True, replace_char=''):
    out = ''
    for word in line.split():
        if str(word).isdecimal() or str(word).isdigit():
            out = out.rstrip() + ' NUMBER '
        else:
            for char in list(word):
                u = ('' + str(char)).encode('utf-8')
                # print(char, '-', len(u))
                if strict_arabic:
                    if len(u) == 2:
                        # print('Char:', u[0], u[1], char)
                        out = out + str(char)
                    else:
                        out = out + replace_char
                else:
                    # this does not really anything.
                    out = out + char
            out = out.rstrip() + ' '
    return out.rstrip()


def chat2csv(filename, output):
    """
    line is in this format: 05/03/2018, 16:19 - <nickname/phone>: <message>
    :param filename:
    :param output:
    :return:
    """
    with open(filename, 'rb') as f_in, \
            open(output, 'w', encoding="utf8", newline='') as f_out:
        for line in f_in:
            # remove invisible chars
            line = line.rstrip()
            # remove the first chars
            if len(line) > 5 and (line[2] == line[5] == 47):
                line = line[20:]
                try:
                    line = line.decode('utf8')
                    # get everything after the first ':'
                    line = line[line.index(':') + 2:]
                except UnicodeDecodeError as ude:
                    pass
                except ValueError as ve:
                    pass

                if str(line) != '<Media omitted>' and str(line) != '<Médias omis>':
                    orig = line
                    line = clean_line(line, replace_char='')
                    if len(line) != 0:
                        # print('Orig:', orig)
                        # print('Curated line:', line)
                        f_out.write(str(line) + '\n')
            else:
                # a new line inside a sentence.
                # ignore
                continue


if __name__ == '__main__':
    chat2csv('./raw_data/WhatsApp Chats/WhatsApp Chat 1.txt', './data/data_output_1.txt')
