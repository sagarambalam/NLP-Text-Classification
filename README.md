# NLP-Text-Classification

## Problem Statement:
This is a Classification problem where we have to classify the text into one of the 12 categories.

## Approach:
1. EDA
    - Cleaning Data. (Tokenizing, Stemming, Stopward removal, Stemming)
    - Additional data cleaning based on data.
2. Building Model.
    - Baseline model using Naive Bayes.
    - Comparing with Random Forest.
3. Hyperparameter tuning.
    - Gridcvsearch
    - Visualising tuning results.
4. Predicting on test dataset.

## Joint Word Removal
1. There are words in the training corpus that contain 2 words joint.
2. This had to be handled in order to not miss out the important vectors.
3. The approach to this problem is by identifying the two words in the words and then splitting at the appropriate length.
4. This has been achieved by using 2 steps
    - package called `wordninja` that does processing at a high speed across the length of the string.
    - in order to tailor the solution to the problem, we use `a custom built lookup table` that is more specific to our data rather than being generic.
    - This look up table has been built by including the words in the `decreasing order of their probability`. Hence based on the highest probable words, the splitting decision is taken

## Predict wrapper function
- `predict.py` file has been built to predict the class given a sentence.
- Usage: 
- `from predict import predict`
- `predict('Input Sentence')`

## Future Scope:
1. Text cleaning based on domain expertise by expanding shortforms.
2. improving approach to split joint words and incorrect spaces
