from parse import conll_to_transitions
import logging
import numpy as np
from nltk import load
import gensim
from sklearn import svm

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

    return vector, transition_mapping[move]


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
    clf = svm.SVC(decision_function_shape='ovr')
    vectors = []
    labels = []
    for transition in gen:
        vector, label = create_vector(transition)
        if vector is None:
            continue
        vectors.append(vector)
        labels.append(label)

    clf.fit(vectors, labels)

    return clf


def main():
    logging.basicConfig(level=logging.INFO)
    set_pos_mapping()
    load_w2v_model()
    clf = train('../data/en.tr100')
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    main()
