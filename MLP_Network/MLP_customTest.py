import torch
from MLP_data import *
from MLP_utils import *
from MLP_model import MLP

def read_test_file(filename):
    with open(filename, 'r') as file:
        data = file.read()
        # Extracting the sentences from data and creating a sentences list []
        sentences = data.strip().split('\n\n')
        processed_sentences = []

        for sentence in sentences:
            # Split the sentence into individual lines (tokens and tags)
            lines = sentence.strip().split('\n')
            # append the lines to the processed_sentences list
            processed_sentences.append(lines)

    return processed_sentences

def load_mlp_model(model_path):
    hidden_layer_1_size = 1024
    hidden_layer_2_size = 4096
    hidden_layer_3_size = 2048

    model = MLP(input_layer_size=1000,
                hidden_layer_1_size=hidden_layer_1_size,
                hidden_layer_2_size=hidden_layer_2_size,
                hidden_layer_3_size=hidden_layer_3_size,
                output_layer_size=44)
    model.load_state_dict(torch.load(model_path))

    return model

def get_predictions(model, features):
    # Perform forward pass
    with torch.no_grad():
        logits = model(features)
        _, predicted_labels = torch.max(logits, 1)

    return predicted_labels

def write_test_file(filename, tagged_sents):
    print("Writing to file...")
    with open(filename, "w") as output:
        for sent in tagged_sents:
            for token in sent:
                output.write(token[0] + ' ' + token[1] + '\n')
            output.write('\n')
    print("Done!")

def main():
    test_file_path   = './test_data.txt'
    model_path       = './save_model/MLP_3layer_200_95.pt'
    data_file_path   = './train.txt'
    output_file_path = './n_grams.test.txt'

    start_tag = 's_tag'
    end_tag = 'e_tag'
    context_length = 2

    _, pos_to_idx  = get_sentences_and_tags(data_file_path)
    sentences      = read_test_file(test_file_path)

    mlp_model   = load_mlp_model(model_path)
    glove_model = load_glove_model()

    print(f"Loaded models.")

    sentences_predicted = []
    num_sent = 0
    for sent in sentences:
        num_sent += 1

        if num_sent%100 == 0:
            print(f"Processing sentence: {num_sent}/{len(sentences)}")

        words_contexts = []
        word_embeddings = []
        for i in range(len(sent)):
            try:
                predecessor, successor = [], []
                if i == 0:
                    predecessor.extend([start_tag, start_tag])
                    for j in range(context_length):
                        successor.extend([sent[i + (j + 1)]])
                elif i == 1:
                    predecessor.extend([start_tag, sent[i - 1]])
                    for j in range(context_length):
                        successor.extend([sent[i + (j + 1)]])
                elif i == len(sent)-2:
                    successor.extend([sent[len(sent)-1], end_tag])
                    for j in range(context_length):
                        predecessor.extend([sent[i - (context_length - j)]])
                elif i == len(sent)-1:
                    successor.extend([end_tag, end_tag])
                    for j in range(context_length):
                        predecessor.extend([sent[i - (context_length - j)]])
                else:
                    for j in range(context_length):
                        predecessor.extend([sent[i - (context_length - j)]])
                        successor.extend([sent[i + (j + 1)]])

                predecessor.extend([sent[i]])
                predecessor.extend(successor)
                word = predecessor

            except Exception as e:
                word = [start_tag, start_tag, start_tag, end_tag, end_tag]

            words_contexts.extend([word])

        for words in words_contexts:
            concat_embeds = []
            concat_embeds.extend([get_glove_word_embedding(glove_model, word) for word in words])
            word_embeddings.append(list(itertools.chain.from_iterable(concat_embeds)))

        # for word in words:
        #     word_embeddings.append(get_glove_word_embedding(glove_model, word))

        word_embeddings_tensor = torch.tensor(word_embeddings)
        predictions = get_predictions(mlp_model, word_embeddings_tensor)

        tags_list = list(pos_to_idx.keys())
        idx_list  = list(pos_to_idx.values())

        k = 0
        word_tags = []
        for pred in predictions:
            predicted_tag = tags_list[idx_list.index(pred)]
            word_tags.append((words_contexts[k][2], predicted_tag))
            k+=1

        sentences_predicted.append(word_tags)

    write_test_file(output_file_path, sentences_predicted)

if __name__ == "__main__":
    main()