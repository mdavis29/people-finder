import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re

_p=(0.3, .7)
_p2=(0.3, .7)

def random_apply(func):
    def apply(*args, **kwargs):
        i = np.random.choice([0,1], 1, p=_p)[0]
        if i == 0:
            return args[0]
        else:
            return func(*args, **kwargs)
    return apply


def random_apply_infreq(func):
    def apply(*args, **kwargs):
        i = np.random.choice([0,1], 1, p=_p2)[0]
        if i == 0:
            return args[0]
        else:
            return func(*args, **kwargs)
    return apply


def error_handler(func):
    def apply(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            print('failure!!!')
            print(func)
            print(*args)
            print(**kwargs)
            raise ValueError('failure')
    return apply

@error_handler
@random_apply
def _aug_seq_insert_double_post(x):
    new_list = x.copy()
    n = len(x)
    i = np.random.choice(range(0, n), 1)[0]
    v = x[i]
    new_list.insert(i, v)
    return new_list

@error_handler
@random_apply
def _aug_seq_pop(x):
    new_list = x.copy()
    n = len(x)
    i = np.random.choice(range(n), 1)[0]
    new_list.pop(i)
    return new_list

@error_handler
@random_apply
def _aug_seq_append_post_zero(x):
    new_list = x.copy()
    new_list.append(0)
    return new_list

@error_handler
@random_apply
def _aug_seq_append_pre_zero(x):
    new_list = x.copy()
    new_list.insert(0, 0)
    return new_list

@error_handler
@random_apply
def _aug_set_random_flip(x):
    new_list = x.copy()
    n = len(x)
    i = np.random.choice(range(0, n-1), 1)[0]
    v1 = new_list[i]
    v2 =  new_list[i+1]
    new_list[i] = v2
    new_list[i+1] = v1
    return new_list

def augment_sequence(x):
    x = _aug_seq_insert_double_post(x)
    x = _aug_seq_pop(x)
    x = _aug_seq_insert_double_post(x)
    x = _aug_seq_pop(x)
    x = _aug_seq_insert_double_post(x)
    x = _aug_seq_pop(x)
    x = _aug_seq_append_post_zero(x)
    x = _aug_seq_append_post_zero(x)
    x = _aug_seq_append_pre_zero(x)
    x = _aug_seq_append_pre_zero(x)
    x = _aug_set_random_flip(x)
    x = _aug_set_random_flip(x)
    x = _aug_set_random_flip(x)
    x = _aug_set_random_flip(x)
    x = _aug_set_random_flip(x)
    x = _aug_seq_insert_double_post(x)
    x = _aug_seq_pop(x)
    x = _aug_seq_insert_double_post(x)
    x = _aug_seq_pop(x)
    x = _aug_seq_insert_double_post(x)
    x = _aug_seq_pop(x)
    x = _aug_seq_append_post_zero(x)
    x = _aug_seq_append_post_zero(x)
    x = _aug_seq_append_pre_zero(x)
    x = _aug_seq_append_pre_zero(x)
    x = _aug_set_random_flip(x)
    x = _aug_set_random_flip(x)
    x = _aug_set_random_flip(x)
    x = _aug_set_random_flip(x)
    x = _aug_set_random_flip(x)
    return x

def _augment_sequences_gen(seq_list):
    for seq in seq_list:
        yield augment_sequence(seq)

def augment_sequences(seq_list):
    g = _augment_sequences_gen(seq_list)
    return list(g)

@error_handler
@random_apply
def _trunc_to_middle_initials(text):
    text_split = text.split(' ')
    n_words = len(text_split)
    try:
        if n_words > 2:
            new_text_list = [text_split[0]]
            for i in range(1, n_words-1):
                new_text_list.append(text_split[i][0])
            new_text_list.append(text_split[-1])
            return ' '.join(new_text_list)
        else:
            return text
    except IndexError:
        return text

@error_handler
@random_apply_infreq
def _add_random_prefix_or_suffix(text):
    suffix = ['Mr.', 'Miss', 'Dr', 'Jr', 'Sr', 'Mrs.', 'phD', 'rev', 'ms']
    s = np.random.choice(suffix, 1)[0]
    if(np.random.choice([1,0], 1)[0] == 1):
        return text + ' ' + s
    else:
        return s + ' ' + text

def _augment_text_gen(text_list):
    for text in text_list:
        yield _add_random_prefix_or_suffix(_trunc_to_middle_initials(text ))

def augment_text(text_list):
    return list(_augment_text_gen(text_list))




def nn_doc_sampler(input_docs, reference_docs):
    c = CountVectorizer(analyzer = 'char')
    ref_x = c.fit_transform(reference_docs)
    input_x = c.transform(input_docs)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(ref_x)
    _, indices = nbrs.kneighbors(input_x)
    index = np.array([v[1] for v in indices])
    return np.array(reference_docs)[index]


@error_handler
def preprocess_text(text):
    text = re.sub('[,]', ' ',text.lower())
    text = re.sub('[.]', ' ',text)
    text = re.sub('  ', ' ',text).strip()
    return text

def _preprocess_text_gen(text_list):
    for text in text_list:
        yield preprocess_text(text)
def preprocess_texts(text_list):
    return list(_preprocess_text_gen(text_list))
