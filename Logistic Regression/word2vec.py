
from gensim.models import Word2Vec
import warnings
 
warnings.filterwarnings(action = 'ignore')

def read_data(data_path):
  sentences  = []
  with open(data_path, 'r',encoding="utf8") as infile:
      sent = []
      for line in infile:
          line = str.split(str.strip(line), ' ')
          if len(line) == 3:
              token, tag_label = line[0], line[1]
              sent.append(token)
              continue
          sentences.append(sent)
          sent = []
  
  return sentences

def main():
  # sentences = read_data('./data/traino.txt')
  # print("-> %d sentences are read from '%s'." % (len(sentences), 'data/pos/train.txt'))
  
  # # #  create a word2vec model
  # model = Word2Vec(sentences, min_count = 1, vector_size = 100,window = 5, sg = 1)

  # # save the model
  # model.save("./model/word2vec.model")
  
  # load the model
  model = Word2Vec.load("./model/word2vec.model")

  # getting the vector of a word
  vector = model.wv['Confidence']

  print("Vector representation of word 'Confidence': \n", vector)

if __name__ == '__main__':
  main()