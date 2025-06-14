
#  CS6910 Assignment 3: Roman to Devanagari Transliteration using Seq2Seq Models

**Student Name**: Raju Guguloth  
**Roll Number**: CS24M036  
**Course**: Deep Learning (DA6401), IIT Madras  
**Instructor**: Prof. Mitesh Khapra  
**Dataset**: Dakshina v1.0  
**W&B Project**: [My_Assignment-3]( https://api.wandb.ai/links/rajunaik-iit-madras/l3c6x7id)  
**Notebook Location**: src/Assignment_DL3 (2).ipynb 
**Predictions**: [Vanilla](predictions_vanilla/predictions_vanilla.csv) | [Attention](predictions_attention/predictions_attention.csv)

#Project: Roman to Devanagari Transliteration using Sequence-to-Sequence Models

# CS6910_Assignment3-master
github:https://github.com/RajuGuguloth/DL_Assignment3



## Problem Statement
This project explores the task of automatic transliteration, specifically converting words written in the Roman script to the Devanagari script. Using the Dakshina dataset, we build and train Sequence-to-Sequence (Seq2Seq) neural network models to learn this mapping.

Goal: To develop a model capable of accurately transliterating Romanized Hindi words into their Devanagari form.

Approach:

We utilize the Seq2Seq architecture, a powerful framework for tasks involving sequence transformations. This involves an encoder to process the input Roman word and a decoder to generate the output Devanagari word.

Steps Taken:

1.Dataset Preparation: Downloaded and extracted the Dakshina dataset for Hindi transliteration. The data was  preprocessed to create character-level or word-level sequences, including tokenization, vocabulary creation, and padding to uniform lengths.
2.Vanilla Seq2Seq Model: Built a baseline Seq2Seq model without an attention mechanism. This model uses RNNs (RNN, GRU, or LSTM) in both the encoder and decoder.
3.Hyperparameter Tuning (Vanilla Model): Conducted experiments with various hyperparameters (e.g., number of layers, hidden size, cell type, dropout) using a Bayesian optimization strategy to find the best-performing vanilla model configuration based on validation accuracy.
4.Seq2Seq Model with Attention: Enhanced the Seq2Seq model by incorporating an attention mechanism. This allows the decoder to focus on relevant parts of the input sequence during output generation.
5.Hyperparameter Tuning (Attention Model): Performed hyperparameter tuning for the attention-based model to optimize its performance.
6.Model Evaluation: Evaluated the best-performing vanilla and attention models on the unseen test set to assess their accuracy and identify types of errors.
7.Prediction Output: Generated and saved the model's predictions on the test set for further analysis.
This project provides a hands-on experience with building, training, and evaluating Seq2Seq models for a practical natural language processing task.

## Requirements

pip install -r requirements.txt


## Steps to run the program
**NOTE :** The program is written in a modular manner, with each logically separate unit of code written as functions.  

- The code is done in a Google colab notebook and stored in the path `src/Assignment3.ipynb` ([link](src/Assignment3.ipynb)). It can be opened and run in Google colab or jupyter server locally.
- The solution to each question is made in the form of a couple of function calls (which are clearly mentioned with question numbers in comments) and commented out in the program so that the user can choose which parts to run and evaluate.
- In order to run the solution for a particular question, uncomment that part and run the cell. Also check the comments in that cell for any cells that need to be run before.
- There are separate functions for creating training model and infefence model, training and sweeping for attention and vanilla models which are clearly mentioned as headings and in comments for the function as well.
- There is a **config** dictionary which is passed to WANDB during training. This is aslo used to create the model in the `seq2seq_no_attention()` and `seq2seq_attention()` functions. The python dictionary contains all hyperparameters and architectural informations for a model. It should contain the following keys :
  ```python
  "learning_rate" --  Learning rate used in gradient descent
  "epochs" --  Number of epochs to train the model
  "optimizer" --  Gradient descent algorithm used for the parameter updation
  "batch_size" --  Batch size used for the optimizer
  "loss_function" --  Loss function used in the optimizer
  "architecture" --  Type of neural network used
  "dataset" --  Name of dataset
  "inp_emb_size" -- Size of input embedding layer
  "no_enc_layers" -- Number of layers in the encoder
  "no_dec_layers" -- Number of layers in the decoder
  "hid_layer_size" -- Size of hidden layer
  "dropout" -- Value of dropout used in the normal and recurrent dropout
  "cell_type" -- Type of cell used in the encoder and decoder ('RNN' or 'GRU' or 'LSTM')
  "beam_width" -- Beam width used in beam decoder
  "attention" -- Whether or not attention is used
  ```
  
      
