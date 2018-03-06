from data import index_samples
import resources as R


def create_dataset():

    def read_file():
        with open(R.SM_DATA) as f:
            return list(set([ tuple(l.replace('\n', '').lower().split('\t')) 
                    for l in f.readlines() ]))

    #vocab = build_vocabulary(max_vocab_size=R.SM_MAX_VOCAB_SIZE)
    lines = read_file()
    texts = [ t for s,t in lines ] 
    sentiments = [ int(s) for s,t in lines ]

    return index_samples(texts, sentiments, max_vocab_size=R.SM_MAX_VOCAB_SIZE)


if __name__ == '__main__':

    dataset = create_dataset()
