from __future__ import absolute_import
from __future__ import print_function

print("importing libraries...")
import sys
import time
import math
import csv
import EvoNN
import FNN
import csv
import warnings
import random
import numpy as np
import evodnn_datasets
warnings.filterwarnings("ignore")
sys.path.append("/Users/Payu/Desktop/EvoNN_package/EvoNN_DNN") # thrid party's libararies, absolute path

"""Load dataset"""
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_wine

from sklearn.metrics import roc_auc_score # determine which of the used models predicts the classes best
from scipy.stats import wilcoxon # test the null hypothesis that the median of a distribution is equal to some value
from sklearn.metrics import mean_squared_error

"""Constant"""
EPS = 1.0E-15
EVOLVER = True
STANDARD = True
ALPHA = 0.01 # define the slope where x < 0 in LReLu

random.seed(1) # Initialize internal state of the random number generator
np.random.seed(1)

if len(sys.argv) > 1:
	dataset_letter = sys.argv[1] # Choose which dataset to use
else:
	dataset_letter = "1"
my_directory = sys.argv[0].replace("tester.py","") # get directory

########################################################
"""Return RMSE of predicted and ground truth"""
def RMSE(y_predicted, y_true):
	rmse = math.sqrt(mean_squared_error(y_true, y_predicted))
	return rmse

##############################################################################
"""Sigmoid Activation Funciton"""
def sigmoid(x):
	return 1/(1+np.exp(-x))

##############################################################################
"""Write benchmark data to .csv file"""
def write_csv_file(filename, the_headers, the_values):
	print("Writing to...",filename)
	the_data_writer, output_file = get_default_CSV_writer(filename)
	the_data_writer.writerow(the_headers)
	for i in range(the_values.shape[0]):
		the_data_writer.writerow(the_values[i])
	if output_file is not None: # Close the written file
		output_file.close()

##############################################################################
def get_default_CSV_writer(filename):
	output_file = open(filename, 'w')
	csv.register_dialect(
                    'mydialect',
                    delimiter = ',',
                    quotechar = '"',
                    doublequote = True,
                    skipinitialspace = True,
                    lineterminator = '\n',
                    quoting = csv.QUOTE_MINIMAL)
	the_data_writer = csv.writer(output_file, dialect='mydialect')
	return the_data_writer, output_file

##############################################################################
"""Shuffle the dataset"""
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

##############################################################################
"""ReLu activation function"""
def ReLU(x):
	y = np.zeros(x.shape) # x has numpy arra
	y[:] = x[:]
	y[y <= 0] = 0 # entries is smaller than zero make them zero
	return y

##############################################################################
"""LReLu activation function"""
def LReLU(x):
	y = np.zeros(x.shape)
	y[:] = x[:]
	y[x <= 0] = ALPHA*x[x <= 0]
	return y

##############################################################################
"""Keep two decimal places"""
def two_dec_digs(x): # numpy array
	y = 100.0*x
	y = np.rint(y)
	y = y / 100.0
	return y

##############################################################################
"""Calculate AUC accuracy
   The closer to -1, the better"""
def myAUC(y_predicted, y_true):
	one_d_y_predicted = y_predicted[:, 0]
	one_d_y_true = y_true[:, 0]
	result = roc_auc_score(one_d_y_true, one_d_y_predicted)
	return -1.0 * result

##############################################################################
"""Multi-class logarithmic loss funciton per class"""
def multiclass_LOGLOSS(y_predicted, y_true):
	logloss_value = 0.0
	for i in range(y_predicted.shape[0]):
		for j in range(y_predicted.shape[1]):
			considered_value = min(max(y_predicted[i][j], EPS), 1.0-EPS)
			logloss_value +=  y_true[i][j]* math.log(considered_value)

	logloss_value *= -(1.0/y_predicted.shape[0])
	return logloss_value

##############################################################################
"""Softmax"""
def softmax(final_layer_values):
	new_layer_values = np.zeros(final_layer_values.shape)
	for i in range(new_layer_values.shape[0]):
		x = final_layer_values[i]
		shiftx = x - np.max(x) # Since we are using an exponent, this is equivalent to dividing. Comes from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
		exponent_layer_values = np.exp(shiftx)
		sum_of_values = np.sum(exponent_layer_values)
		new_layer_values[i] = exponent_layer_values/sum_of_values
	return new_layer_values

##############################################################################
"""
if (dataset_letter == "a"):
	print("Loading iris...")
	dataset = load_iris()
elif (dataset_letter == "b"):
	print("Loading breast cancer...")
	dataset = load_breast_cancer()
elif (dataset_letter == "c"):
	print("Loading diabetes...")
	dataset = load_diabetes()
elif (dataset_letter == "d"):
	print("Loading digits...")
	dataset = load_digits()
elif (dataset_letter == "e"):
	print("Loading wine...")
	dataset = load_wine()
else:
	print("Select datasets")
	exit()
"""

if (dataset_letter == '1'):
	"""
	attributes: 8
	instances: 4177
	"""
	X, Y = evodnn_datasets.load_abalone()
elif (dataset_letter == '2'):
	X, Y = evodnn_datasets.load_ecoli()
elif (dataset_letter == '3'):
	X, Y = evodnn_datasets.load_haberman()
elif (dataset_letter == '4'):
	X, Y = evodnn_datasets.load_ilpd()
elif (dataset_letter == '5'):
	X, Y = evodnn_datasets.load_lymphography()
elif (dataset_letter == '6'):
	X, Y = evodnn_datasets.load_mammographic()
elif (dataset_letter == '7'):
	X, Y = evodnn_datasets.load_spect()
elif (dataset_letter == '8'):
	X, Y = evodnn_datasets.load_spectf()
elif(dataset_letter == '9'):
	X, Y = evodnn_datasets.load_statlog()
elif (dataset_letter == '10'):
	X, Y = evodnn_datasets.load_thoracic()
elif (dataset_letter == '11'):
	X, Y = evodnn_datasets.load_tumor()
elif (dataset_letter == '12'):
	X, Y = evodnn_datasets.load_wilt()
elif (dataset_letter == '13'):
	X, Y = evodnn_datasets.load_yeast()
else:
	print("Select datasets")
	exit()


##############################################################################
"""X is feature map, Y is ground truth"""
#X = dataset.data
#Y = dataset.target

"""Normlize the dataset in the range [-1, 1]"""
for i in range(X.shape[1]):
	largest_value = np.amax(X[:, i])
	smallest_value = np.amin(X[:, i])
	if (largest_value == 0):
		if (smallest_value != 0):
			X[:, i] /= abs(smallest_value)
	else:
		X[:, i] /= max(largest_value, abs(smallest_value))

"""Set macro and initilization"""
evo_pairs = []
standard_pairs = []

evo_logloss = []
standard_logloss = []

shuffle_number = 5 #20 # column; config here to reduce running time
rep_number = 3 #10 # row; config here to reduce running time

evo_runtimes = []
standard_runtimes = []

evo_headers = []
standard_headers = []
evo_measurements = np.zeros((rep_number, shuffle_number))
standard_measurements = np.zeros((rep_number, shuffle_number))

for a in range(shuffle_number):

	evo_headers.append("shuffle #"+str(a+1))
	standard_headers.append("shuffle #"+str(a+1))
	X, Y = shuffle_in_unison(X, Y)

	sample_size = Y.shape[0]
	#class_array = ['a','b','d','e','1']
	#if (dataset_letter in class_array):
	myType = 'classification'
	class_number = np.max(Y)+1 # number of classification
	Y_classes = np.zeros((sample_size, class_number))
	for i in range(sample_size):
		j = Y[i]
		Y_classes[i][j] = 1.0 # ground truth
	"""
	else:
		myType = 'regression'
		largest_value = np.amax(Y[:])
		smallest_value = np.amin(Y[:])
		Y_classes = Y/max(largest_value, abs(smallest_value)) # ground truth
	"""
	print("X is a",X.shape[0],"X",X.shape[1],"matrix")
	if (len(Y_classes.shape) == 2):
		print("Y is a",Y_classes.shape[0],"X",Y_classes.shape[1],"matrix")
	else:
		print("Y is a",Y_classes.shape[0],"X 1 matrix")

	train_index = int(sample_size*0.6) # 60% trainning data
	validate_index = int(sample_size*0.8) # 20% validation data

	# Separate to train, valid, and test set
	X_train, Y_train = X[:train_index], Y_classes[:train_index]
	X_valid, Y_valid = X[train_index:validate_index], Y_classes[train_index:validate_index]
	X_test, Y_test = X[validate_index:], Y_classes[validate_index:]

	for i in range(rep_number):
		myRep = i+(a*rep_number)
		if (EVOLVER == True):
			start_time = time.process_time()
			if (myType == 'classification'):
				final_function = softmax
				fitness = multiclass_LOGLOSS
				if (dataset_letter == 'e'):
					fitness = myAUC
			elif (myType == 'regression'):
				final_function = two_dec_digs
				fitness = RMSE

			myEvoNNEvolver = EvoNN.Evolver(	G=1000,						# Maximum iteration
											early_stopping=100,				# Minimum iteration, no use for now
											node_per_layer = [ 20, 20, 20, 20, 20],		# Number of nodes per layer
											MU=50,							# Number of parents
											LAMBDA=50,						# Number of offspring
											P_m=0.01,						# Weight mutation probability
											P_mf=0.01,						# Function mutation probablity
											R_m=0.1,						# Weight mutation radius
											P_c=0.3,						# Crossover proportion
											P_b=0.01,						# Bias mutation probability
											R_b=0.1,						# Bias mutation radius
											elitism=True,					# Elitism involves copying a small proportion of the fittest candidates, unchanged, into the next generation.
											tournament_size=2,				# Selecting individuals from a population
											fitness_function=fitness,
											final_activation_function=final_function,
											additional_functions=[LReLU, ReLU],
											random_state=myRep,
											verbose=0)
			myEvoNNEvolver.fit(X_train, Y_train, X_valid, Y_valid) # Train the net

			if (myType == 'classification'):
				print("This is classification.")
				evo_Y_pred_probabilities = myEvoNNEvolver.predict_proba(X_test)
				if (dataset_letter == 'e'):
					evo_my_logloss = myAUC(evo_Y_pred_probabilities, Y_test)
				else:
					evo_my_logloss = multiclass_LOGLOSS(evo_Y_pred_probabilities, Y_test)
			else:
				print("This is regression.")
				evo_Y_pred_probabilities = myEvoNNEvolver.predict(X_test)
				evo_my_logloss = RMSE(evo_Y_pred_probabilities, Y_test)

			elapsed_time = time.process_time() - start_time
			evo_runtimes.append(elapsed_time)
			evo_logloss.append(evo_my_logloss) # the lower the better

			# benchmark
			evo_logloss_arr = np.array(evo_logloss)
			evo_logloss_mean = np.mean(evo_logloss_arr)
			evo_logloss_std = np.std(evo_logloss_arr)
			evo_measurements[i][a] = evo_my_logloss
			evo_avg = np.mean(evo_runtimes)
			evo_SD = np.std(evo_runtimes)

			print("evo logloss {} +/- {}".format(evo_logloss_mean, evo_logloss_std))
			print(evo_logloss_arr)
			print("evo NN runtime per cycle = {} +/- {}".format(evo_avg, evo_SD))

		if (STANDARD == True):
			start_time = time.process_time()
			net_fnn = FNN.Network(
				sizes=[X_train.shape[1], 20, 20, 20, 20, 20, Y_train.shape[1]], # number of neurons in the respecitve layer of the network
				type=myType
			)

			# format data represntation
			training_inputs = [np.reshape(x, (X_train.shape[1], 1)) for x in X_train]
			training_results = [np.reshape(y, (Y_train.shape[1], 1)) for y in Y_train]
			training_data = list(zip(training_inputs, training_results))

			validation_inputs = [np.reshape(x, (X_valid.shape[1], 1)) for x in X_valid]
			validation_results = [np.reshape(y, (Y_train.shape[1], 1)) for y in Y_valid]
			validation_data = list(zip(validation_inputs, validation_results))

			test_inputs = [np.reshape(x, (X_test.shape[1], 1)) for x in X_test]
			test_results = [np.reshape(y, (Y_test.shape[1], 1)) for y in Y_test]
			test_data = list(zip(test_inputs, test_results))

			"""Train the network"""
			net_fnn.SGD(training_data=training_data,
						epochs=1000,
						early_stopping=100,
						mini_batch_size=10,
						eta=1.0,							# learning rate
						test_data=validation_data,
						test_true=Y_valid,
						verbose=1
			)

			if (myType == 'classification'):
				standard_Y_pred_probabilities = net_fnn.predict_proba(test_data)
				"""
				if (dataset_letter == 'e'):
					standard_my_logloss = myAUC(standard_Y_pred_probabilities, Y_test)
					if (standard_my_logloss > -0.5):
						standard_my_logloss = -1.0 - standard_my_logloss

				else:
			    """
				standard_my_logloss = multiclass_LOGLOSS(standard_Y_pred_probabilities, Y_test)
			else:
				standard_Y_pred_probabilities = net_fnn.predict_proba(test_data)
				for i in range (standard_Y_pred_probabilities.shape[0]):
					max_index = np.argmax(standard_Y_pred_probabilities[i])
					standard_Y_pred_probabilities[i][:] = 0
					standard_Y_pred_probabilities[i][max_index] = 1.0
				standard_my_logloss = RMSE(standard_Y_pred_probabilities, Y_test)

			elapsed_time = time.process_time() - start_time
			standard_runtimes.append(elapsed_time)
			standard_logloss.append(standard_my_logloss)

			standard_logloss_arr = np.array(standard_logloss)
			standard_logloss_mean = np.mean(standard_logloss_arr)
			standard_logloss_std = np.std(standard_logloss_arr)
			standard_measurements[i][a] = standard_my_logloss
			standard_avg = np.mean(standard_runtimes)
			standard_SD = np.std(standard_runtimes)

			print("standard logloss {} +/- {}".format(standard_logloss_mean, standard_logloss_std))
			print(standard_logloss_arr)
			print("standard NN runtime per cycle = {} +/- {}".format(standard_avg, standard_SD))

if (STANDARD == True) and (EVOLVER == True):
	W, p_value = wilcoxon(x = evo_logloss_arr,
							y = standard_logloss_arr)
	print("p_value is {}".format(p_value))

if (STANDARD == True):
	standard_headers = np.array(standard_headers)
	standard_filename = my_directory+dataset_letter+"_standard_results.csv"
	write_csv_file(standard_filename, standard_headers, standard_measurements)

if (EVOLVER == True):
	evo_headers = np.array(evo_headers)
	evo_filename = my_directory+dataset_letter+"_evo_results.csv"
	write_csv_file(evo_filename, evo_headers, evo_measurements)

print("Done")
