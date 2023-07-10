###################################################################################
#   GitHub implementation of MLP. Used to check performance with Julia
#	Classifier for MNIST datasets. Julia has better accuracy but slower performance
#   @Author: Unknown @Developer (modified): Thomas	@Date:29 Dec 2018
###################################################################################
import numpy as np

import time

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

###################################################################################
#				Global constants for the Artificial Neural Network (ANN)
#		Model - MLP - Multi-Layer Perceptron. One Hidden Layer. Hand-coded (no lib)
###################################################################################


inputs = 785

outputs = 10

train_size = 60000

test_size = 10000

total_epochs = 10

learning_rate = 0.01

###################################################################################
# Loading csv files into memory - train data, train labels, test data, test labels
###################################################################################
def file_load(filename, n_rows):
    
	data = np.loadtxt(filename, delimiter=',')

	# scaling and normalizing data
	dataset = np.insert(data[:n_rows, np.arange(1, inputs)]/255, 0, 1, axis=1)

	labels = data[:n_rows, 0]

	return dataset, labels

train_data, train_labels = file_load('mnist_train.csv', train_size)

test_data, test_labels = file_load('mnist_test.csv', test_size)

###################################################################################
#                            Neural Network Functions
###################################################################################

# Activation function
def sigmoid(x):
        return (1/(1 + np.exp(-x)))


def dot_product(data):
	output_array = np.dot(np.reshape(data, (1, inputs)), weights)					
	return sigmoid(output_array)


def error_calculation(output_array, predicted):
	y = np.insert(np.zeros((1, outputs-1)), np.argmax(output_array), 1)				
	t = np.insert(np.zeros((1, outputs-1)), predicted, 1)					
	return t-y
	
def update_weights(error, data, weights):
	delta = np.dot(np.reshape(data, (inputs, 1)), np.reshape(error, (1, outputs)))
	weights += (learning_rate * delta) # Updated weights
	return weights

def training_perceptron(weights):
	for i in range(0, train_size):
		output_array = dot_product(train_data[i, :])				      
        # Feed-forward an image sample to get output array

		error = error_calculation(output_array, int(train_labels[i]))		
        # Evaluate to find array representation of predicted output

		weights = update_weights(error, train_data[i, :], weights)		    
        # Back propagate error through the network to get adjusted weights

	return weights
	
def testing_perceptron(dataset, labels, size):

	predicted_output = []

	for i in range(0, size):

		output_array = dot_product(dataset[i, :])				    
        # Feed-forward an image sample to get output array

		predicted_output.append(np.argmax(output_array))				
        # Append the predicted output to correct_output list

	return accuracy_score(labels, predicted_output), predicted_output


###################################################################################
#                  Run the ANN at global level - convenience
###################################################################################
start = time.time()
print("Starting now:\n")

# Randomize Weights are generated in the range of (-0.05,0.05)
weights = (np.random.rand(inputs, outputs) - 0.5)*(0.1)
               
previous_accuracy = 1
epoch = 0

while True:
        # Accuracy is calculated on training dataset
        current_accuracy, predicted_output = testing_perceptron(train_data, train_labels, train_size)

        print("\nEpoch " + str(epoch) + ":\nTraining Accuracy = " + str(current_accuracy))

        # Accuracy is calculated on test dataset
        test_accu, predicted_output = testing_perceptron(test_data, test_labels, test_size)			
       
        print("Test Accuracy = " + str(test_accu) + "\n")
        previous_accuracy = current_accuracy
        
        epoch += 1
        
        # Train the network
        weights = training_perceptron(weights)        
        
        if epoch >= total_epochs:
                break

###################################################################################
# 					      Output Metrics and Reports
###################################################################################
     
# Testing the network on test set, calculating testing accuracy                                                               
test_accu, predicted_output = testing_perceptron(test_data, test_labels, test_size)				
print("\nFinal Test Accuracy = " + str(test_accu) + 
      "\nLearning Rate = " + str(learning_rate) + 
	  "\nConfusion Matrix :\n")

#calculating confusion matrix depicting how many correct output is classified for the testing data           
print(confusion_matrix(test_labels, predicted_output))           

#classification_report                        
print("\nClassification Report\n")
print(classification_report(test_labels, predicted_output))
print("\n")

end = time.time()
print("Time taken: " + str(end - start) + " seconds")

###################################################################################
# ------------------------------------E-O-F----------------------------------------
###################################################################################