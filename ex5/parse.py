from collections import deque


# given a sentence, returns a list of transitions
def conll_to_transitions(sentence):

    s = []  #stack
    b = deque([])  #buffer

    transitions = []

    for w in sentence:
        b.append(w)

    s.append(['0', 'ROOT', '_', '_', '_', '_', '0', '_', '_', '_'])

    while len(b) > 0 or len(s) > 1:
        if s[-1][0] == '0':   # the root
            add_shift(s, b, transitions)
        elif s[-2][6] == s[-1][0] and check_rightest_arc(s[-2], b):
            add_left(s, b, transitions)
        elif s[-1][6] == s[-2][0] and (len(b) == 0 or s[-2][0] != '0') and check_rightest_arc(s[-1], b):
            add_right(s, b, transitions)
        elif len(b) == 0:
            print "Non projective"
            return None
        else:
            add_shift(s, b, transitions)
    return transitions


def check_rightest_arc(word, b):
    for w in b:
        if w[6] == word[0]:
            return False
    return True


def add_shift(s, b, transitions):
    word = b.popleft()
    top2 = None
    if len(s) > 1:
        top2 = s[-2]
    transitions.append([s[-1], top2, word, 'SHIFT'])
    s.append(word)


def add_left(s, b, transitions):
    top1 = s.pop()
    top2 = s.pop()
    transitions.append([top1, top2, b[0] if len(b) > 0 else None, 'LEFT'])
    s.append(top1)


def add_right(s, b, transitions):
    top1 = s.pop()
    top2 = s.pop()
    transitions.append([top1, top2, b[0] if len(b) > 0 else None, 'RIGHT'])
    s.append(top2)




def main():
    input_file = "/Users/kfirbar/Downloads/p2_code+data/en.tr100"
    h = open(input_file, 'r')
    sentence = []
    for l in h:
        l = l.strip()
        if l == "":
            trans = conll_to_transitions(sentence)
            sentence = []
        else:
            sentence.append(l.split('\t'))



if __name__ == "__main__":
    main()