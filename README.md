# NN-CA4-1.Image-Captioning-2.Intent-Classification

### 1.Intent Classification [Link](#part-1-intent-classification)

### 2.Image Captioning [Link](#part-2-image-captioning)

# Part 1: Intent Classification

In this report, we present the implementation of code for intent classification in question-answering using LSTM architectures based on the research article [Intent Classification in Question-Answering Using LSTM Architectures](https://arxiv.org/pdf/2001.09330.pdf).

### 2. Data Preprocessing

#### Normalization

One of the crucial steps in text data preprocessing is normalization. In this phase, the input text is transformed into a standard form. This may involve removing punctuation marks, converting letters to lowercase, or eliminating the use of specific words or characters. Important normalization methods include:

- Removal of punctuation marks.
- Conversion of letters to lowercase.
- Expansion of abbreviations to full words.

#### Tokenization

Tokenization involves dividing the input text into smaller units, typically words, but they can also include phrases, numbers, and punctuation marks. Tokens are often used as processing units for various text processing algorithms. Key tokenization methods include:

- Tokenization based on whitespace.
- Tokenization based on punctuation.
- Tokenization using regular expressions.
- Tokenization based on parts-of-speech (POS) tags.

Each of these methods can be used for text preprocessing, depending on the type of text and the algorithm being applied. Some essential text preprocessing techniques include:

- Stop word removal.
- Stemming.
- Lemmatization.
- Named Entity Recognition (NER).
- Chunking.
- Sentiment analysis.
- Machine translation.

In this report, a dataset with 5452 samples for training (df_train) and 500 samples for testing (df_test) is used. Each dataset contains three columns: coarse_label, fine_label, and text. The coarse_label represents the main class, and the fine_label represents the sub-class. There are six unique values in coarse_label, namely ABBR, ENTY, DESC, HUM, LOC, and NUM, along with 47 sub-classes.

After applying the mentioned preprocessing steps, sentences are embedded. To achieve this, words in each sentence are first tokenized, padded to a specified length (e.g., 50), and then converted into 300-dimensional vectors using pre-trained word embeddings such as GloVe.

### 3. Model Implementation and Results

In the paper, the accuracy achieved on the training set is 99.26%, while in this project, it reached 98.98%. For the validation set, the paper reports an accuracy of 87.80%, whereas this project achieved 95.15%, demonstrating better results. The size_hidden parameter was tuned, and a size of 100 performed better compared to 25, as it increased model complexity and improved classification accuracy for main classes. However, an excessively large size_hidden can lead to longer training times and overfitting.

Comparison of Model Accuracy (Size_Hidden: 25 vs. 100):

| Size_Hidden | Paper Training Set (%) | Paper Test Set (%) | Project Training Set (%) | Project Test Set (%) |
| ----------- | ---------------------- | ------------------ | ------------------------ | -------------------- |
| 25          | 99.67                  | 96.74              | 99.26                    | 87.80                |
| 100         | 98.98                  | 95.15              | 99.98                    | 90.20                |

### 4. Responder Model Implementation

The implemented model architecture includes a Bidirectional LSTM layer with 100 units for size_hidden. The model employs a softmax activation function and Mean Squared Error (MSE) as the loss function.

The generated answers are not exact but can provide insights into the content's general nature. The quality of the answers depends on various factors and is an area for further improvement.

# Part 2: Image Captioning

In this section, we present the implementation of an image captioning model based on the research article [Image Captioning](https://arxiv.org/pdf/1805.09137.pdf).

### Model with Frozen CNN

**Overview**

In this part, we load and freeze the ResNet-18 model for feature extraction and caption generation.

**Data Preprocessing**

To begin, we download the dataset and implement functions for loading and preprocessing it:

- Lowercase conversion
- Punctuation removal
- Elimination of extra whitespaces
- Removal of single-letter words
- Appending the "endseq" token to captions

**Captioning Approach**

Since our model uses the LSTM network, which only receives image features once, we've made some interesting choices:

- Omitting the "startseq" token to avoid delaying meaningful word generation
- Leveraging post-padding to ensure uniform sentence lengths
- Using only the "endseq" token

**Model Architecture**

We load ResNet-18, omitting the default final linear layer, and use a custom linear layer for feature embedding. Dropout is applied after the LSTM layer.

**Training and Results**

Training these models can be time-consuming, so we've implemented a Learning Rate Scheduler for stability. We split the dataset into training and test sets and create data generators for both.

**Model Outputs**

Despite having more parameters, this model exhibits higher training error and poor performance on the test data.

### Model with Trainable CNN

**Overview**

In this section, we make the ResNet model trainable.

**Challenges**

- Training CNN from scratch alongside the untrained network can be challenging
- CNN might not learn the desired task effectively

**Training and Results**

Despite having more parameters, this model exhibits higher training error and poor performance on the test data.

---

Please refer to the project files for detailed code and results. This README provides a high-level overview of our image captioning implementation.
