
## Introduction
In this practice we are going to train several Deep Learning Architectures for Sequence Processing including:
- lstm
- rnn
- gru
- stacked_lstm
- stacked_gru
- bidirectional_lstm
- cnn_unigram
- cnn_bigram
- cnn_unibigram

### Prepare the datasets and models
1.  In this practice, we are going to use the word2vec functioanlity from GloVe. 
Therefore, you need to download the `glove.840B.300d.txt` from here:
    - Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): [glove.840B.300d.zip](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip) [[mirror](https://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip)]
    Once you downloaded the zip file, extract the `glove.840B.300d.txt` into the `data/` folder.
1.  The dataset that we are going to train our models on is [Text Document Classification Dataset](https://www.kaggle.com/datasets/sunilthite/text-document-classification-dataset?resource=download) from kaggle.
    - Download the datset, which gives you a zip file `df_file.csv.zip`
    - Extract this zip file and add the `df_file.csv` file to the `data/` folder.

Now your `data` folder should looks like this:  
📦data  
 ┣ 📜df_file.csv  
 ┗ 📜glove.840B.300d.txt  

 