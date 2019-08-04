class Indexer:
    def __init__(self, elements):
        self._element_to_index = {"<UNKNOWN>": 0}

        for x in elements:
            if x not in self._element_to_index:
                self._element_to_index[x] = len(self._element_to_index)

        self._index_to_element = {v: k for k,v in self._element_to_index.items()}

    def get_element_to_index_dict(self):
        return self._element_to_index

    def element_to_index(self, element):
        return self._element_to_index.get(element, 0)

    def index_to_element(self, index):
        return self._index_to_element[index]

    def elements_to_index(self, elements):
        return [self.element_to_index(x) for x in elements]

    def indexes_to_elements(self, indexes):
        return [self.index_to_element(x) for x in indexes]

    def size(self):
        return len(self._element_to_index)
