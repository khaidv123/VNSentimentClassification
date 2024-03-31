import torch


def get_word_vectors_loaded():
    word_vectors_loaded = torch.load('word_embeddings.pt')
    return word_vectors_loaded

#print(word_vectors_loaded.vectors.shape)
# ==> torch.Size([6250, 100])

def get_vector_for_a_word(embeddings, word):
    """ Get embedding vector of the word
    @param embeddings (torchtext.vocab.vectors.Vectors)
    @param word (str)
    @return vector (torch.Tensor)
    """
    assert word in embeddings.stoi, f'*{word}* is not in the vocab!'
    return embeddings.vectors[embeddings.stoi[word]]


#print(get_vector(get_word_vectors_loaded(), "chào"))

"""
tensor([-0.0386,  0.1077,  0.0133,  0.0368,  0.0314, -0.1502,  0.1115,  0.2369,
        -0.1344, -0.0281, -0.0246, -0.1728, -0.0492,  0.0541, -0.0089, -0.0785,
         0.0274, -0.1344, -0.0203, -0.1570,  0.0407,  0.0124, -0.0002, -0.1022,
        -0.0346,  0.0444, -0.0924, -0.0911, -0.0461, -0.0119,  0.1332,  0.0067,
         0.0519, -0.0555, -0.0260,  0.1644,  0.0390, -0.0886, -0.0556, -0.1981,
         0.0091, -0.0994, -0.0472,  0.0529,  0.1183,  0.0048, -0.1217, -0.0362,
         0.0729,  0.0579,  0.0557, -0.1333, -0.0405,  0.0253, -0.1411,  0.0600,
         0.0810, -0.0507, -0.1446,  0.0290, -0.0011,  0.0544, -0.0320, -0.0234,
        -0.1347,  0.0862,  0.0239,  0.0986, -0.0980,  0.1047, -0.1935,  0.0666,
         0.1097, -0.0889,  0.1432,  0.0583, -0.0046,  0.0781, -0.1732,  0.0425,
        -0.0140,  0.0315, -0.1292,  0.1299, -0.0074,  0.0192,  0.0197,  0.0801,
         0.1184,  0.0093,  0.1033,  0.0667,  0.0174, -0.0071,  0.1448,  0.0625,
         0.0249, -0.0687,  0.0813, -0.0031])
"""

