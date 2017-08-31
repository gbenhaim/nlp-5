from parse import conll_to_transitions
import logging
import numpy as np
from nltk import load
import gensim
import nn

LOGGER = logging.getLogger(__name__)

POS_TAG_IDX = 4
WORD_IDX = 1

transition_mapping = {
    'SHIFT': 0,
    'LEFT': 1,
    'RIGHT': 2,
    0: 'SHIFT',
    1: 'LEFT',
    2: 'RIGHT'
}

N_CLASSES = len(transition_mapping) / 2

pos_mapping = None
NUM_OF_POS = 0

w2v_model = None

def set_pos_mapping():
    global pos_mapping
    global NUM_OF_POS
    reverse_mapping = {}
    pos_mapping = load('help/tagsets/upenn_tagset.pickle')
    NUM_OF_POS = len(pos_mapping) + 1
    for i, pos in enumerate(pos_mapping):
        pos_mapping[pos] = i
        reverse_mapping[i] = pos

    pos_mapping['_'] = NUM_OF_POS - 1
    pos_mapping[NUM_OF_POS - 1] = '_'
    pos_mapping.update(reverse_mapping)
    LOGGER.debug('pos mapping {}'.format(pos_mapping))


def load_w2v_model():
    LOGGER.info('loading w2v model')
    global w2v_model
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../w2v/GoogleNews-vectors-negative300.bin', binary=True)
    LOGGER.info('finished loading w2v model')


def create_pos_vector(pos):
    v = np.zeros(NUM_OF_POS, dtype=np.float64)
    v[pos_mapping[pos]] = 1
    return v


def create_class_one_hot(move):
    v = np.zeros(N_CLASSES)
    idx = transition_mapping[move]
    v[idx] = 1
    return v


def create_vector(transition):

    w1 = transition[0]
    w2 = transition[1]
    move = transition[3]

    try:
        w1_vec = w2v_model[w1[WORD_IDX]]
    except (KeyError, TypeError) as e:
        LOGGER.debug(w1)
        LOGGER.debug(e.message)
        return None, None

    try:
        w2_vec = w2v_model[w2[WORD_IDX]]
    except (KeyError, TypeError) as e:
        LOGGER.debug(w2)
        LOGGER.debug(e.message)
        return None, None

    vector = np.hstack(
        (
            w1_vec.astype(np.float64),
            w2_vec.astype(np.float64),
            create_pos_vector(w1[POS_TAG_IDX]),
            create_pos_vector(w2[POS_TAG_IDX])
         )
    )

    return vector, create_class_one_hot(move)


def sentence_gen(source_file):
    with open(source_file, mode='rt') as f:
        sentence = []
        line = f.readline()
        while line:
            while line != '\n':
                line = line.strip()
                sentence.append(line.split('\t'))
                line = f.readline()

            yield sentence
            line = f.readline()
            sentence = []


def transition_gen(data_path):
    gen = sentence_gen(data_path)
    for sentence in gen:
        transitions = conll_to_transitions(sentence)
        if not transitions:
            continue
        for transition in transitions:
            yield transition


def train(data_path):
    LOGGER.info('Building classifier')
    gen = transition_gen(data_path)
    vectors = []
    labels = []
    for transition in gen:
        vector, label = create_vector(transition)
        if vector is None:
            continue
        vectors.append(vector)
        labels.append(label)

    n_features = len(vectors[0])
    n_classes = len(labels[0])

    LOGGER.debug('number of feature = {}'.format(n_features))
    LOGGER.debug('number of classes = {}'.format(n_classes))

    return nn.train((vectors, labels), n_features, n_classes)


def main():
    logging.basicConfig(level=logging.INFO)
    set_pos_mapping()
    load_w2v_model()
    print train('../data/en.tr100')


if __name__ == '__main__':
    main()
