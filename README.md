# Midterm-NGrams

## Overview:

The repo contains the implementation of Hidden Markov Model, Logistic Regression, and Multi-Layer Perceptron for POS tagging.
The dataset used for training is:  https://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz 

Format of the training file:

&lt;Word/Token&gt;   &lt;POS-TAG&gt;  &lt;Chunking Tag&gt;

**Note: For the implementation of the POS tagging we are only using the first two columns of the training file.

## Models:
1. Hidden Markov Model (HMM): 
>>The implementation is done using a dynamic programming algorithm named Viterbi. The model resulted in test accuracy of approx. 90% on the test split for vanilla version of Viterbi algorithm and was able to attain a accuracy of approx. 95% for modified Viterbi (for regular expressions). Though we expect this model to perform as out best model but as HMM is based on the computation of emission & transition probabilies and we are dealing with 44 classes (or tags as per the conll2000 dataset) therefore, each run was taking minimum of 7-8hrs. Due to limited compute resources the generation of output from the unlabelled test data took unexpected longer to generate. 

2. Logistic Regression (LR):
>>The logistic regression model has been implemented using the scikit-learn pre-built machine learning library. The model was trained using pre-trained Glove embeddings of 50 dimensions. The current accuracy of the model is approximately 84.33%, without considering any local contexts for calculating the word embeddings in the training dataset. However, it has been observed that if appropriate context and features are taken into consideration, the accuracy of the model can be increased to 96%.

3. Multi Layer Perceptron (MLP):
>>The MLP implementation for POS tagging was done using a 3-hidden layer network. The features were input by converting each word in embeddings using the GloVe model. Each word embedding was a 25-dimensional vector and the labels for each of these vectors we one-hot encoded with 44 classes. The network was trained using Cross Entropy Loss and Stochastic Gradient Descent optimizer. The initial classification output was about 90%. The network was further improved using the windowing technique where for any given word two predecessors and two successors were considered to bring context into the picture. In addition, the word embedding vector dimension was increased to 200. With these additions, we were able to achieve an accuracy of 95.71% on the validation set and 95.41% on the testing dataset. This model achieved the best accuracy among the three algorithms we tried.

## Best model:
Based on the results our best model is **Multi Layer Perceptron (MLP)** with validation accuracy as **95.71%** and test accuracy as **95.41%**.


## Labelled Test Data:
The file named 'n_grams.text.txt' contains the output from the best model (MLP).
