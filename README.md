# CS4680---Assignment-A1

## Project Overview

This project demonstrates a simple machine learning workflow to predict Ice Cream Revenue based on Temperature using a linear regression model from scikit-learn.

**Feature:** Temperature  
**Target:** Ice Cream Revenue

## Dataset

The dataset is downloaded from [Kaggle: Ice Cream Revenue Data](https://www.kaggle.com/datasets/vinicius150987/ice-cream-revenue/data). The dataset consists of 500 rows which is sufficiently enough for training the model.

Below are the first 10 rows of the dataset:

| Temperature | Revenue     |
|-------------|-------------|
| 24.56688442 | 534.7990284 |
| 26.00519115 | 625.1901215 |
| 27.79055388 | 660.6322888 |
| 20.59533505 | 487.7069603 |
| 11.50349764 | 316.2401944 |
| 14.35251388 | 367.9407438 |
| 13.70777988 | 308.8945179 |
| 30.83398474 | 696.7166402 |
| 0.976869989 | 55.39033824 |
| 31.66946458 | 737.8008241 |

## Model Choice

By visualizing the data, we observe a linear relationship between Temperature and Revenue. Therefore, a linear regression model is chosen for prediction.

Additionally, a Decision Tree regression model is also implemented for comparison. The Decision Tree model is a non-linear model that can fit the training data very closely.

**Note:**
If we do not split the dataset into training and testing sets, and instead train and test the Decision Tree model on the same data, the model will predict the target values exactly (i.e., the predictions will be identical to the actual revenue values in the dataset). This is because the Decision Tree can memorize the training data, leading to perfect predictions on seen data, but this does not reflect real-world performance. To properly evaluate the model, it is important to split the dataset into separate training and testing sets.