from training_utils import *
import datetime


if __name__ == '__main__':
    # this training took 45min on my computer which is pretty good for a computer using a cpu.
    trained_model = train_model_on_main_dataset()
    print('The model is trained --------------------------------------')
    model_filename = 'model_' + datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S") + '.bin.gz'
    # save the model
    save_full_model(trained_model, './models/' + model_filename)

    print('Loading model')
    # loaded_model = load_full_model('./models/model_binary_2019-01-21-120832.h5')
    # loaded_model = load_full_model('./models/model_2019-01-22-184506.bin.gz') #model_binary_2019-01-22-090325.h5')
    loaded_model = load_full_model('./models/' + model_filename)
    print('Model loaded')
    evaluate_model_vectorization(loaded_model, './datasets/test/questions.txt')
    evaluate_model_similarity(loaded_model, './datasets/test/distances.txt', margin=0., verbose=True)
