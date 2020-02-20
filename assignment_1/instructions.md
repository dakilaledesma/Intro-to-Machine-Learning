# Assignment 1 Instructions

This assignment is easy and relatively straightforward:
## 1.
You are given two text files: ```english.txt``` and ```german.txt``` within this folder. Use the text files in order to build your training dataset and your labels (answers).

## 2.
Get another language that is not English or German and add that language as part of your dataset. For your labels, if English is 0, and German is 1, then your third language that you decide to add must be represented by another number.

## 3.
Sift the dataset down to only 5-letter words. Pre-process your data using whatever means you would like; you don't have to use ```ord()```, for example, if you have a better idea.

## 4. 
Make a training and testing dataset split *yourself*. In the tutorial, you were given this code block:

```
# Predicting “ANYONE”
print(knn_model.predict([[65, 78, 89, 79, 78, 69]])) # Output: 0 (English)
print(svm_model.predict([[65, 78, 89, 79, 78, 69]])) # Output: 0 (English)
print(mlp_nn.predict([[65, 78, 89, 79, 78, 69]])) # Output: 0 (English)

# Predicting “BÄRGET”
print(knn_model.predict([[66, 196, 82, 71, 69, 84]])) # Output: 1 (German)
print(svm_model.predict([[66, 196, 82, 71, 69, 84]])) # Output: 1 (German)
print(mlp_nn.predict([[66, 196, 82, 71, 69, 84]])) # Output: 1 (German)

# Note that predict() must also take a 2D array as our training data was a 2D array.
```

However, you would've noticed in the tutorial that we were *testing the same words that we were training the models with*. This is not helpful, as the model can merely "memorize" the word and would get it right. We want to use the model in order to predict words that it has *never seen or been trained with before*. As such, split your training dataset into two arrays. The first is your actual training dataset and should include 80% of the total words used. The second dataset is the remaining 20% *in a separate array*. You'll be using this array to test your models accuracy on words it was not trained with.

Randomly sample when splitting your datasets! If you sample only the last 20% for your testing dataset, you may only get words that start with the last few letters, for ex. words that start with T to Z.  If you sample only the first 20% for your testing dataset, you may only get words that start with the first few letters, for ex. words that start with A to E. 

## 5. 
Train your models using your created train dataset, and test your models using your test dataset. You need to do all 3 of the models, (KNN, SVM, and MLP) and have them predict on all three of the languages (English, German, and a language of your choice).

## 6.
Graph your results in the same way cited in the tutorials.


# Grading rubric
**Out of 100 points**
- 40 points: Making of a proper dataset with 3 languages, proper processing of data for training the models.
- 10 points: Making a training and testing dataset split.
- 30 points: Being able to train the models and make predictions. 
- 10 points: Graph the results of each model (KNN, SVM, MLP) as seen in the tutorial
- 10 points: On your worst performing model, for every accuracy point below 65%, lose 1 point.
  - For example, if your worst performing model gets 62%, you will get a 3 point deduction.
