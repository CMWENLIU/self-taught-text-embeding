import numpy as np
import re
import glob, os
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    '''
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]
	'''
    #data_sets = glob.glob("./data/*.txt")
    data_sets = "./data/"
    #store all of the class data in a list
    class_data = []
    label_list = []
    default_list = []

    #load data from files
    for i in os.listdir(data_sets):
        print (data_sets+i)
        examples = list(open(data_sets+i).readlines())
        examples = [s.strip() for s in examples]
        #append these examples to the list of lists
        class_data.append(examples)
        #make the label list as long as the numbe rof classes
        default_list.append(0)

    # concat class examples
    counter = 0
    for class_examples in class_data:
        #set the label
        temp_list = [0] * len(class_data)
        temp_list[counter] = 1
        label_list.append(temp_list)
        if counter == 0:
            x_text = class_examples
        else:
            x_text = x_text + class_examples
        counter += 1

    #clean and split
    x_text = [clean_str(sent) for sent in x_text]
    #x_text = [s.split(" ") for s in x_text]

    # Generate labels
    final_labels = []
    counter = 0
    for class_examples in class_data:
        print (label_list[counter])
        final_labels.append([label_list[counter] for _ in class_data[counter]])
        counter += 1

    y = np.concatenate(final_labels, 0)
    return [x_text, y]

def load_embedding_vectors(vocabulary, filename, vector_size):
    # load embedding_vectors from the embedding file
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
