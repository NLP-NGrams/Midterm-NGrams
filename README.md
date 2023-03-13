# Midterm-NGrams

The repo contains the implementation of Hidden Markov Model, Logistic Regression, and Multi-Layer Perceptron for POS tagging.  
The dataset used for training is: https://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz

Format of the training file:

Word/Token POS-TAG Chunking Tag

**Note: For the implementation of the POS tagging we are only using the first two columns of the training file.

Models:

1. Hidden Markov Model : 

The implementation is done using a dynamic programming algorithm named Viterbi. The model resulted in test accuracy of approx. 90% on the test split for vanilla version of Viterbi algorithm and was able to attain a accuracy of approx 95% for modified Viterbi (for regular expressions). Though we expect this model to perform as out best model but as HMM is based on the computation of emission & transition probabilies and we are dealing with 44 classes (or tags as per the conll2000 dataset) therefore, each run was taking minimum of 7-8hrs. Due to limited compute resources the generation of output from the unlabelled test data took unexpected longer to generate.

2. Logistic Regression:

The logistic regression model has been implemented using the scikit-learn pre-built machine learning library. The model was trained using pre-trained Glove embeddings of 50 dimensions. The current accuracy of the model is approximately 84.33%, without considering any local contexts for calculating the word embeddings in the training dataset. However, it has been observed that if appropriate context and features are taken into consideration, the accuracy of the model can be increased to 96%.

3. Multi Layer Perceptron:

With that said our best model is MLP with accuracy as: 95.03%

Labelled Test Data:

The file named 'n_grams.text.txt' contains the output from the best model (MLP).