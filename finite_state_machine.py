import numpy as np

class Grammar():
    def __init__(self, edges, edge_to_value):
        """ Grammar has to be non-ambiguous (deterministic)"""
        self.edges = edges
        self.edge_to_value = edge_to_value
        self.value_to_edge = {}
        for edge, value in self.edge_to_value.items():
            if value in self.value_to_edge:
                self.value_to_edge[value] += edge
            else:
                self.value_to_edge[value] = [edge]

def in_grammar(seq, G):
    """checks if the sequence is in the given Grammar"""
    curr_node = 0 # sequences always start at 0
    for el in seq:
        possible_edges = [(curr_node, target) for target in G.edges[curr_node]]
        if -1 in np.ndarray.flatten(np.array(possible_edges)):
            # since the Grammar has multiple transition with the same letter, it's possible to have
            # messed up more than one step ago and not notice it.
            return False
        possible_letters = [G.edge_to_value[edge] for edge in possible_edges]
        if (el not in possible_letters):
            # current element wasn't possible to obtain from previous node
            return False
        else:
            # find next node based on taken edge
            # check which edge has el as its end
            real_edge = filter(lambda x: G.edge_to_value[x] == el, possible_edges)
            if len(real_edge) > 1:
                print "YOUR GRAMMAR IS AMBIGUOUS"
            else:
                real_edge = real_edge[0]

        curr_node = real_edge[1]

    return True

def generate_sequence(G, out_of_grammar=False):
    """
    Generate sequence from grammar :G:,
                        TODO: not sure how much to deviate by
                      OR
                      out of grammar :G: by a small deviation (1 symbol)
    """

    string = ""
    curr_node = 0

    sample_next_node = lambda curr_node: G.edges[curr_node][np.random.randint(0, len(G.edges[curr_node]))] # uniform choice across all possible edges
    next_node = sample_next_node(curr_node)

    while next_node != -1:
        taken_edge = (curr_node, next_node)
        curr_node = next_node
        string += G.edge_to_value[taken_edge]
        next_node = sample_next_node(curr_node)

    if out_of_grammar:
        # generate ouf of grammar deviation
        success = False
        letters = list(set(G.edge_to_value.values())) # possible letters we could have assigned
        new_string = None
        while not success:
            new_string = string
            index = np.random.randint(0, len(new_string))

            # choose a different letter than is already in place
            substitute_letter = letters[np.random.randint(0, len(letters))]
            while substitute_letter == new_string[index]:
                substitute_letter = letters[np.random.randint(0, len(letters))]

            # put substitute letter where original one was
            new_string = new_string[0:(index-1)] + substitute_letter + new_string[index:]

            # success if the altered new_string is not in grammar :G:
            success = not in_grammar(new_string, G)

        # done, assign altered string to the original
        string = new_string

    # add beginning and end
    string = "B" + string + "E"
    return string



# Reber's Grammar
n = 6 # number of nodes
edges = {0: [1, 3], 1: [1, 2], 2: [3, 5],
         3: [3, 4], 4: [2, 5], 5:[-1]} # Reber's grammar, -1 signals the end of sequence
edge_to_value = {(0,1): "T",
              (0,3): "P",
              (1,2): "X",
              (1,1): "S",
              (2,3): "X",
              (2,5): "S",
              (3,3): "T",
              (3,4): "V",
              (4,2): "P",
              (4,5): "V"}
G = Grammar(edges, edge_to_value)
unique_chars = list(set(G.edge_to_value.values()))  + ['B', 'E']

# make dataset
def generate_grammar_dataset(N):
    """"
    generates :N: sequeces for both test/train half in grammer half out
    generate both test and train simultaneously so that we can pad both test
    and train to the longest sequence in both.
    """
    unique_chars = list(set(G.edge_to_value.values()))  + ['B', 'E'] # plus begining and end symbols
    # convert everything to indeces range(len(unique_chars))
    char_to_index = {letter: i for i, letter in enumerate(unique_chars)}
    convert_to_indices = lambda seq: [char_to_index[el] for el in seq]

    x_train = [generate_sequence(G) for i in range(N/2)]
    x_test = [generate_sequence(G) for i in range(N/2)]
    y_train = [1 for i in range(N/2)]
    y_test = [1 for i in range(N/2)]
    x_train += [generate_sequence(G, out_of_grammar=True) for i in range(N / 2)]
    x_test += [generate_sequence(G, out_of_grammar=True) for i in range(N / 2)]
    y_train += [0 for i in range(N / 2)]
    y_test += [0 for i in range(N / 2)]

    # pad all sequences to be as long as the longest sequence generated
    merged_set = x_train + x_test
    maxlen = len(max(x_train + x_test, key=lambda x: len(x)))
    for i, seq in enumerate(merged_set):
        # pad with "B"'s in the beginning
        seq = "B"*(maxlen - len(seq)) + seq
        merged_set[i] = seq
    x_train, x_test = merged_set[0:N], merged_set[N:]

    # convert only after all padding is done
    x_train_indices = [convert_to_indices(seq) for seq in x_train]
    x_test_indices = [convert_to_indices(seq) for seq in x_test]

    x_train_one_hot = embed_one_hot(x_train_indices, len(unique_chars))
    x_test_one_hot = embed_one_hot(x_test_indices, len(unique_chars))
    return x_train, \
           np.expand_dims(y_train, axis=1), \
           x_test, \
           np.expand_dims(y_test, axis=1), \
           x_train_one_hot, \
           x_test_one_hot, \
           maxlen, \
           unique_chars


def embed_one_hot(batch_array, depth=len(unique_chars)):
    """
    :batch_array: gets shaped into one-hot vectors of :depth:
    * :batch_array: needs to start being indexed from 0 for its elements.
    """
    batch_size, length = len(batch_array), len(batch_array[0])
    if batch_size == 0.0:
        batch_size = batch_array.shape[0]

    one_hot_matrix = np.zeros((batch_size, length, depth))
    for i,array in enumerate(batch_array):
        one_hot_matrix[i, np.arange(len(array)), array] = 1
    return one_hot_matrix


x_train, y_train, x_test, y_test, x_train_one_hot, x_test_one_hot, maxlen, unique_chars = generate_grammar_dataset(2)
