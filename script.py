# libraries
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# KMeans parameter
n_clusters= 17
k_fold = 10

# to find mean
def find_mean(x):
	return np.mean(x)

# to find standard deviation
def find_stadard_deviation(x):
	return np.std(x)

# reading dataset "board_games.csv"
board_games = pd.read_csv("board_games.csv")

# removing unknown value rows
board_games_dropna = board_games.dropna(axis = 0)

# removing rows having (average_rate = 0) from board_games
board_games_filtered = board_games_dropna[board_games_dropna["average_rating"] != 0]

# getting rows having (average_rate = 0) for testing purpose
board_games_testset = board_games_dropna[board_games_dropna["average_rating"] == 0]

# getting average_rating column
average_rating = board_games_filtered["average_rating"]
average_rating = np.array(average_rating)

# getting histogram of average_rating
average_rating_histogram = np.histogram(average_rating)

# making histogram graph of average_rating_histogram
plt.hist(average_rating_histogram)
plt.show()

# calculating standard deviation and mean of average_rating
stddev_average_rating = np.std(average_rating)
mean_average_rating = np.mean(average_rating)

print(stddev_average_rating)
print(mean_average_rating)

# getting column names
column_names = board_games_filtered.columns.values
column_names = column_names[3:]

# preparing kmeans data from board_games_filtered for KMeans
numeric_columns = board_games_filtered[column_names]

# getting kmeans classifier
kmeans = KMeans(n_clusters = n_clusters)

# fitting numeric_columns into kmeans classifier
kmeans.fit(numeric_columns)

# extracting labels_ attribute of kmeans
labels = kmeans.labels_

# getting mean of each row of numeric_columns
game_mean = numeric_columns.apply(find_mean, axis = 1)

# getting standard deviation of each row of numeric columns
game_std = numeric_columns.apply(find_stadard_deviation, axis = 1)

# plotting a graph
plt.scatter(x = game_mean, y = game_std, c = labels)
plt.show()

# finding correlations among columns of dataframe
correlations = numeric_columns.corr(method = 'pearson')

# getting "average_rating" columun from correlations
corr_average_rating = correlations["average_rating"]
corr_average_rating = pd.DataFrame(corr_average_rating)

# sorting corr_average_rating in descending order
sorted_corr = corr_average_rating.sort_values(["average_rating"], ascending = [False])
print(sorted_corr)

# "bayes_average_rating" is created by using average_rating.
# So, taking other top two column names as predictors, i.e, 
# average_weight and minage.
predictors = numeric_columns[["average_weight", "minage"]]

# getting values of target variable -> "average_rating"
target = numeric_columns["average_rating"]
target = pd.DataFrame(target)

# getting LinearRegression model
reg = linear_model.LinearRegression()

# object for cross validation
cross_valid = cross_validation.KFold(len(predictors), n_folds = k_fold)

# training LinearRegression model for 10 folds
for train_indices, test_indices in cross_valid:
	trainset_x, testset_x = predictors.iloc[train_indices], predictors.iloc[test_indices]
	trainset_y, testset_y = target.iloc[train_indices], target.iloc[test_indices]
	
	# training LinearRegression model
	reg.fit(trainset_x, trainset_y)
	
	# getting training accuracy
	training_accuracy = reg.score(testset_x, testset_y)
	
	print("Training Accuracy : " + str(training_accuracy))

# getting test set
predictors = board_games_testset[["average_weight", "minage"]]

# testing LinearRegression on test set
predictions = reg.predict(predictors)
predictions = pd.DataFrame(predictions)

# concatenating predictors and their corresponding predictions
predictors = predictors.reset_index()
predictors = predictors.drop("index", axis = 1)
predictions = predictions.reset_index()
predictions = predictions.drop("index", axis = 1)
output = pd.concat([predictors, predictions], axis = 1)

# saving predicted values with their independent variable in a csv file
column_names = output.columns.values
column_names = column_names.tolist()
column_names = column_names[:-1]
column_names.append("predicted_average_rating")
output.columns = column_names
output.to_csv("output.csv")