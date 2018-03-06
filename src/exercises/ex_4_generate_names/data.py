import resources as R
import os


def read_files():
    """
    TODO

    Args : 

    Returns :

    """

    def read_file(filename):
        with open(filename) as f:
            return [ line.replace('\n', '') for line in f.readlines() ]

    return [ read_file(os.path.join(R.DATA, l)) for l in R.LANG_FILES ]

def create_dataset():
    """
    TODO

    Args:

    Returns:

    """
    list_of_names = read_files()
    samples = []
    for i, names in enumerate(list_of_names):
        samples.extend([ (name, i) for name in names ])

    return index_samples(samples)

def build_vocabulary(samples):
    """
    TODO

    Add 0 for padding

    Args:

    Returns:

    """
    return [ '0' ] + sorted(set(''.join([ sample[0] for sample in samples ])))

def index_samples(samples):
    """
    TODO

    Args:

    Returns:

    """
    vocab = build_vocabulary(samples)
    ch2i  = { ch:i for i,ch in enumerate(vocab) }
    return {
            'raw_samples' : samples,
            'samples'     : [ ([ ch2i[ch] for ch in name ], label) for name, label in samples ], 
             'vocab'      : vocab,
             'ch2i'       : ch2i
             }
