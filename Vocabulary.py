import torch
from underthesea import word_tokenize

class Vocabulary:
    """ The Vocabulary class is used to record words, which are used to convert
        text to numbers and vice versa.
    """
    def __init__(self):
        self.word2id = dict()
        self.word2id['<pad>'] = 0   # Pad Token
        self.word2id['<unk>'] = 1   # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return len(self.word2id)

    def id2word(self, word_index):
        return self.id2word[word_index]

    def add(self, word):
        if word not in self:
            word_index = self.word2id[word] = len(self.word2id)
            self.id2word[word_index] = word
            return word_index
        else:
            return self[word]

    @staticmethod
    def tokenize_corpus(corpus):
        tokenized_corpus = list()
        for document in corpus:
            tokenized_document = [word.replace(" ", "_") for word in word_tokenize(document)]
            tokenized_corpus.append(tokenized_document)

        return tokenized_corpus

    def corpus_to_tensor(self, corpus, is_tokenized=False):
        if is_tokenized:
            tokenized_corpus = corpus
        else:
            tokenized_corpus = self.tokenize_corpus(corpus)
        indicies_corpus = list()
        for document in tokenized_corpus:
            indicies_document = torch.tensor(list(map(lambda word: self[word], document)),
                                             dtype=torch.int64)
            indicies_corpus.append(indicies_document)

        return indicies_corpus

    def tensor_to_corpus(self, tensor):
        corpus = list()
        for indicies in tensor:
            document = list(map(lambda index: self.id2word[index.item()], indicies))
            corpus.append(document)

        return corpus
    



def indices2embeddings(tensor, word_embedding):
    for sentence_tensor in tensor:
        sentence_embeddings = []
        for index in sentence_tensor:
            if index < len(word_embedding.vectors) and index >= 2:
                word_embed = word_embedding.vectors[index-2]
                sentence_embeddings.append(word_embed)
        sentence_embeddings_tensor = torch.stack(sentence_embeddings)
    return sentence_embeddings_tensor
    



#corpus_sample = ["Với cộng đồng người Bách Việt trước đây, việc thuần hóa mèo cũng có thể theo cách thức như vậy =)).",
#                 "Tuy nhiên, rất khó xác định được thời gian cụ thể loài mèo được thuần hóa.😎",
#                 "Chỉ biết rằng, từ xa xưa, mèo đã là vật nuôi thân quen trong hầu hết gia đình nông dân Việt Nam."]


#print(Vocabulary.tokenize_corpus(corpus_sample))
"""
Tokenize the corpus...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 
[00:00<00:00, 17.78it/s]
[['Với', 'cộng_đồng', 'người', 'Bách_Việt', 'trước_đây', ',', 'việc', 'thuần_hóa', 'mèo', 'cũng', 'có_thể', 'theo', 'cách_thức', 
'như_vậy', '=))', '.'], ['Tuy_nhiên', ',', 'rất', 'khó', 'xác_định', 'được', 'thời_gian', 'cụ_thể', 'loài', 'mèo', 'được', 
'thuần_hóa', '.', '😎'], ['Chỉ', 'biết', 'rằng', ',', 'từ', 'xa_xưa', ',', 'mèo', 'đã', 'là', 'vật_nuôi', 'thân_quen', 'trong', 
'hầu_hết', 'gia_đình', 'nông_dân', 'Việt_Nam', '.']]

"""


#vocab = Vocabulary()

# create vocabulary from pretrained word2vec
#words_list = list(get_word_vectors_loaded().stoi.keys())
#for word in words_list:
#    vocab.add(word)


#print(vocab.__len__())
# ==> 6252


# Test vocab
    
#corpus_to_tensor
#tensor = vocab.corpus_to_tensor(corpus_sample)
#print(tensor)

#Tokenize the corpus...
#100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 20.40it/s] 
#100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<?, ?it/s] 
#[tensor([ 980,    1,   19,    1,    1,   46,   64,    1, 1797,   16,    1,  482,
#           1,    1,  199,  114]), tensor([   1,   46,  125,  263,    1,   38,    1,    1, 3157, 1797,   38,    1,
#         114, 2219]), tensor([ 863,   62,  772,   46,   67,    1,   46, 1797,   60,    2,    1,    1,
#          81,    1,    1,    1,    1,  114])]


#tensor_to_corpus
#corpus = vocab.tensor_to_corpus(tensor)
#print(corpus)


# Lưu trữ đối tượng vocab 
#with open('vocab.pkl', 'wb') as f:
#    pickle.dump(vocab, f)





