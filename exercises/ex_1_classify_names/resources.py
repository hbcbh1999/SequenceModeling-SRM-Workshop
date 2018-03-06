import os


# data directory
DATA='../../data/names/'

LANG_FILES = sorted(os.listdir(DATA))
# list of languages
lang = [ lf.replace('.txt', '') for lf in LANG_FILES ]
# language to index lookup
lang2i = { l:i for l,i in enumerate(lang) }
