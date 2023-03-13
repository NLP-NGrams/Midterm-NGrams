import torch
import os.path
import itertools
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

## This class to create torch Dataset
class create_dataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_tensor = self.features[idx].clone().detach()
        label_tensor = self.labels[idx].clone().detach()
        return feature_tensor, label_tensor

# Load the Glove model to get word embeddings
def load_glove_model():
    if os.path.isfile('pretrained_embeds/gensim_glove_vecto.txt'):
        glove_model = KeyedVectors.load_word2vec_format("pretrained_embeds/gensim_glove_vectors.txt", binary=False)
    else:
        glove2word2vec(glove_input_file="pretrained_embeds/glove.twitter.27B.200d.txt", word2vec_output_file="pretrained_embeds/gensim_glove_vectors.txt")
        glove_model = KeyedVectors.load_word2vec_format("pretrained_embeds/gensim_glove_vectors.txt", binary=False)

    return glove_model

# Given a word, get its embedding
def get_glove_word_embedding(glove_model, word):
    word = word.lower()
    try:
        return glove_model.get_vector(word)
    except Exception as e:
        # print(f"Glove Exception: {e}")
        return glove_model.get_vector('unk')

# Given sentences and tags, create a torch dataset with one-hot encoding for MLP
def get_dataset(sentences, pos_to_idx, glove_model):
    tags_words, tags, word_embeddings= [], [], []

    # Create a list of all the words and tags
    for pair in sentences:
        tags_words.extend([(tag, word) for (word, tag) in pair])

    # Get embeddings and labels for (tag, word) pair
    for (tag, word) in tags_words:
        tags.append(pos_to_idx[tag])
        word_embeddings.append(get_glove_word_embedding(glove_model, word))

    # Convert word embeddings to tensors and labels to one-hot
    word_embeddings = torch.tensor(word_embeddings, dtype=torch.float32)
    tag_one_hot = F.one_hot(torch.tensor(tags), num_classes=len(pos_to_idx))

    # Create a torch dataset
    dataset = create_dataset(word_embeddings, tag_one_hot.float())

    return dataset

def get_context_dataset(sentences, pos_to_idx, context_length, glove_model):
    tags_words, tags, word_embeddings= [], [], []

    start_tag = 's_tag'
    end_tag   = 'e_tag'

    # Create a list of all the words and tags
    for sent in sentences:
        for i in range(len(sent)):
            try:
                predecessor, word, successor = [], [], []
                if i == 0:
                    predecessor.extend([start_tag, start_tag])
                    for j in range(context_length):
                        successor.extend([sent[i + (j + 1)][0]])
                elif i == 1:
                    predecessor.extend([start_tag, sent[i - 1][0]])
                    for j in range(context_length):
                        successor.extend([sent[i + (j + 1)][0]])
                elif i == len(sent)-2:
                    successor.extend([sent[len(sent)-1][0], end_tag])
                    for j in range(context_length):
                        predecessor.extend([sent[i - (context_length - j)][0]])
                elif i == len(sent)-1:
                    successor.extend([end_tag, end_tag])
                    for j in range(context_length):
                        predecessor.extend([sent[i - (context_length - j)][0]])
                else:
                    for j in range(context_length):
                        predecessor.extend([sent[i - (context_length - j)][0]])
                        successor.extend([sent[i + (j + 1)][0]])

                predecessor.extend([sent[i][0]])
                predecessor.extend(successor)
                word = predecessor
                tag  =  sent[i][1]

            except Exception as e:
                word = [start_tag, start_tag, start_tag, end_tag, end_tag]
                tag  = '.'

            tags_words.extend([(tag, word)])

    # Get embeddings and labels for (tag, word) pair
    for (tag, words) in tags_words:
        concat_embeds = []
        concat_embeds.extend([get_glove_word_embedding(glove_model, word) for word in words])
        word_embeddings.append(list(itertools.chain.from_iterable(concat_embeds)))
        tags.append(pos_to_idx[tag])

    # Convert word embeddings to tensors and labels to one-hot
    word_embeddings = torch.tensor(word_embeddings, dtype=torch.float32)
    tag_one_hot = F.one_hot(torch.tensor(tags), num_classes=len(pos_to_idx))

    # Create a torch dataset
    dataset = create_dataset(word_embeddings, tag_one_hot.float())

    return dataset

# Read the data file and fetch sentences and tags
def get_sentences_and_tags(filename):
    sentences = []
    with open(filename, 'r') as infile:
        sent = []
        for line in infile:
            line = str.split(str.strip(line), ' ')
            if len(line) == 3:
                token, tag_label = line[0], line[1]
                sent.append((token, tag_label))
                continue
            sentences.append(sent)
            sent = []

    # Create a dataframe to fetch the tags and their positions
    df = pd.DataFrame(sum(sentences, []), columns=['word', 'pos'])
    df['pos_idx'] = df['pos'].astype('category').cat.codes
    pos_to_idx = {k: v for (v, k) in enumerate(df['pos'].astype('category').cat.categories)}

    return sentences, pos_to_idx


# Create dataloaders from the sentences and their positions
def get_dataloaders(glove_model, sentences, pos_to_idx, batch_size=32, val_frac=0.1, test_frac=0.1):

    # Split the dataset into training, validation and testing
    train_sentences, test_sentences = train_test_split(sentences, test_size=test_frac)
    train_sentences, val_sentences = train_test_split(train_sentences, test_size=val_frac/(1-test_frac))

    # Create torch datasets
    train_dataset  = get_context_dataset(train_sentences, pos_to_idx, 2, glove_model)
    val_sentences  = get_context_dataset(val_sentences, pos_to_idx, 2, glove_model)
    test_sentences = get_context_dataset(test_sentences, pos_to_idx, 2, glove_model)

    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_sentences, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_sentences, batch_size=batch_size, shuffle=True)

    # Confirm dimensions
    for batch_idx, (feature, labels) in enumerate(train_loader):
        print(f"Feature batch dims: {feature.shape}")
        print(f"Feature label dims: {labels.shape}")
        print(f"Class labels example (one-hot encoded): {labels[:2]}")
        break

    return train_loader, val_loader, test_loader