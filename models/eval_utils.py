import Levenshtein as Lev


def char_to_word(output_list):
    word_string = "".join(output_list)
    return word_string


def wer(s1, s2):
    s1 = char_to_word(s1)
    s2 = char_to_word(s2).strip()

    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]

    score = Lev.distance(''.join(w1), ''.join(w2)) / len(s2.split())

    return score


def cer(s1, s2):
    s1 = char_to_word(s1)
    s2 = char_to_word(s2).strip()

    word_s1, word_s2, = s1.replace(' ', ''), s2.replace(' ', '')

    score = Lev.distance(word_s1, word_s2) / len(s2)

    return score
