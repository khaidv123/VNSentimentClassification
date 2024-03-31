from src.Vocabulary import Vocabulary
from src.RNNmodel import RNN
import torch
import gradio as gr
import torchtext.vocab as vocab1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

word_embedding = vocab1.Vectors(name = "word2vec.txt",
                               unk_init = torch.Tensor.normal_)


vocab = Vocabulary()


words_list = list(word_embedding.stoi.keys())
for word in words_list:
    vocab.add(word)

INPUT_DIM = word_embedding.vectors.shape[0]
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.3
PAD_IDX = vocab["<pad>"]
UNK_IDX = vocab["<unk>"]

model = RNN(INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            PAD_IDX)
model.embedding.weight.data.copy_(word_embedding.vectors)
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

model.to(device)

model.load_state_dict(torch.load('Best_model_RNN.pt', map_location=torch.device('cpu')))

def predict_sentiment(model, sentence, vocab, device):
    model.eval()
    corpus = [sentence]
    tensor = vocab.corpus_to_tensor(corpus)[0].to(device)
    tensor = tensor.unsqueeze(1)
    length = [len(tensor)]
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

def predict(text):
  pre = predict_sentiment(model, text, vocab, device)
  return {'Positive':(pre),
          "Negative": (1- (pre))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.components.Textbox(label='Input query'),
    outputs=gr.components.Label(label='Predictions'),
    allow_flagging='never'
)


demo.launch(share=True)
