import gensim
from os import walk
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
    Save the given model as keyed vectors model into the given path
    :param model: a trained model
    :param path: the file path
    :return: nothing
    """
    # model.save(path)
    gensim.models.KeyedVectors.save_word2vec_format(model.wv, path, binary=True)


def load_model(path):
    """
    Loads a keyed vectors model from the given path
    :param path: the file containing the saved model
    :return: the model, if the file does not exist you will notice.
    """
    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True, encoding='utf8')


def save_full_model(model, path):
    """
    Save the given model as trainable into the given path
    :param model: a trained model
    :param path: the file path
    :return: nothing
    """
    model.save(path)


def load_full_model(path):
    """
    Loads a trainable model from the given path
    :param path: the file containing the saved model
    :return: the model, if the file does not exist you will notice.
    """
    return gensim.models.Word2Vec.load(path)


def online_train(model, more_sentences):
    """
    Trains a given model on new data.
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


def read_file(path):
    # read data
    data = []
    with open(path, 'rb') as f_in:
        for line in f_in:
            if len(line.split()) > 5:
                data.append(line.decode('utf8').rstrip())
    return data


def single_train_model(datafile):
    # train a new model
    start = time.time()
    sentences = read_file(datafile)
    print('Training model with %s sentences' % len(sentences))
    model = train_model(sentences, epochs=150, vect_size=25)
    end = time.time()
    print('Model training took %s seconds.' % (end - start))
    return model


def train_and_save_model():
    # train a new model
    sentences = read_file('./data/data_output.txt')  # 'here comes the new comers\nand the second winner is'.split('\n')
    print(sentences)
    model = train_model(sentences)
    print(get_embeddings(model))
    # save it to a tmp file
    save_model(model, './models/simple_model_binary.h5')
    # load it again
    loaded_model = load_model('./models/simple_model.h5')
    print('loading model')
    loaded_model = gensim.models.KeyedVectors.load_word2vec_format('./models/simple_model_binary.h5', binary=True, encoding='utf8')
    print('model loading')
    # print the word embeddings
    print(get_embeddings(loaded_model))
    # train it on more data
    more_sentences = ['good', 'this', 'is']
    loaded_model = online_train(loaded_model, more_sentences)
    # print the new embeddings
    print(get_embeddings(loaded_model))
    words_in = ['محمد', 'هههه', 'فيديو']
    for word_in in words_in:
        print(word_in)
        for word in loaded_model.wv.most_similar(word_in, topn=15):
            print(word)


def evaluate_model_vectorization(model, test_data_filename):
    """
    Evaluates the answers given by the model against the given ones from the file.
    The questions should be a four words lines: question1 answer1 question2 answer2. This is similar to
    'man king women queen' classical test.
    :param model: The trained model
    :param test_data_filename: The file with the questions
    :return: Nothing
    """
    # load test file
    with open(test_data_filename, "rb") as infile:
        for line in infile:
            line = line.rstrip()
            if chr(line[0]) == ':':
                # skip
                continue
            else:
                q1, a1, q2, a2 = line.decode('utf8').split()
                print('%s->%s, %s->?' % (q1, a1, q2), model.wv.most_similar(positive=[a1, q2], negative=[q1], topn=5))


def evaluate_model_distance(model, test_data_filename, margin=0, verbose=False):
    """
    Evaluate the performance of the given model based on the evaluated similarity (or 1-distance) between given words.
    The test is to check if the distance between the query word and the positive word is smaller than the distance
     between the query word and the negative word.
    :param model: The trained model to evaluate
    :param test_data_filename: the file location with the distance query,
    each line contains three word: a query word <space> a positive word <space> a negative word
    :param margin: if set, the distance has to be greater than this param to count for a correct answer
    :param verbose: If True the result for each line is printed
    :return: The overall accuracy of the model on this evaluation(the ratio of correct answers over the number of tests)
    """
    count = 0
    correct_answers = 0
    # load test file
    with open(test_data_filename, "rb") as infile:
        for line in infile:
            line = line.rstrip()
            if chr(line[0]) == ':':
                # skip
                continue
            else:
                word, positive, negative = line.decode('utf8').split()
                d_pos, d_neg = model.wv.similarity(word, positive), model.wv.similarity(word, negative)
                count = count + 1
                if d_pos - d_neg > margin:
                    if verbose:
                        print('Correct (Pos: %f, Neg: %f)' % (d_pos, d_neg))
                    correct_answers = correct_answers + 1
                else:
                    if verbose:
                        print('Incorrect (Pos: %f, Neg: %f)' % (d_pos, d_neg))
    print("Distances accuracy:%s%%" % (100 * correct_answers / count))


def train_model_on_main_dataset():
    train_folder = './datasets/train/'
    model = None
    for _, _, files in walk(train_folder):
        for file in files:
            print('Training model on file:', train_folder + file)
            if model is None:
                model = single_train_model(train_folder + file)
            else:
                model = online_train(trained_model, read_file(train_folder + file))

    return model


if __name__ == '__main__':
    trained_model = train_model_on_main_dataset()
    model_filename = 'model_binary_' + datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S") + '.h5'
    # save the model
    save_full_model(trained_model, './models/' + model_filename)
    print('The model is now trained --------------------------------------')

    print('Loading model')
    # model = load_full_model('./models/model_binary_2019-01-21-120832.h5')
    # model = load_full_model('./models/model_binary_2019-01-22-090325.h5')
    loaded_model = load_full_model('./models/' + model_filename)
    print('Model loaded')
    # evaluate_model_vectorization(model, './datasets/test/questions.txt')
    evaluate_model_distance(loaded_model, './datasets/test/distances.txt', margin=0.3)
