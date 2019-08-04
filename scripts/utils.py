DOCSTART_LITERAL = '-DOCSTART-'


def read_ner_data_from_connl(path_to_file):
    words = []
    tags = []

    with open(path_to_file, 'r', encoding='utf-8') as file:
        for line in file:
            splitted = line.split()
            if len(splitted) == 0:
                continue
            word = splitted[0]
            if word == DOCSTART_LITERAL:
                continue
            entity = splitted[-1]
            words.append(word)
            tags.append(entity)
        return words, tags


def get_batched(words, labels, size):
    for i in range(0, len(labels), size):
        yield (words[i:i + size], labels[i:i + size])


def load_embedding_dict(vec_path):
    from tqdm import tqdm
    import numpy as np
    embeddings_index = dict()

    with open(vec_path, 'r', encoding='UTF-8') as file:
        for line in tqdm(file.readlines()):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    return embeddings_index


def get_tag_indexes_from_scores(scores):
    import numpy as np
    predicted = []
    for i in range(scores.shape[0]):
        predicted.append(int(np.argmax(scores[i])))
    return predicted
