using Distributions
using DelimitedFiles
using LinearAlgebra
using MLDatasets

# Activation function
relu(x) = x * (x > 0)
#sigmoid(x) = 1 / (1 + exp(-x))
mutable struct NetInfo
    # Input Layer, Hidden Layer and Output Layer nodes numbers
    inodes::Int
    hnodes::Int
    onodes::Int

    # Link weight matrix
    wih::Array{Float64, 2} # from input layer to hidden layer
    who::Array{Float64, 2} # from hidden layer to output layer
    # learning rate
    lr::Float64
end
# Factory method - Init Net
function InitNet(inodes::Int, hnodes::Int, onodes::Int, lr::Float64)
    wih = rand(Normal(0, hnodes^(-0.5)), hnodes, inodes)
    who = rand(Normal(0, onodes^(-0.5)), onodes, hnodes)
    return NetInfo(inodes, hnodes, onodes, wih, who, lr)
end
# Query Result
function Query(net::NetInfo, inputs::Array)
    # Calculate the signal entering the hidden layer
    hidden_inputs = net.wih * inputs
    hidden_outputs = relu.(hidden_inputs)

    # Calculate the signal entering the output layer
    final_inputs = net.who * hidden_outputs
    final_outputs = relu.(final_inputs)

    return final_outputs
end

function Train!(net::NetInfo, inputs::Array, targets::Array)
    # PART 1: Consistent with the Query function
    # Calculate the signal entering the hidden layer
    hidden_inputs = net.wih * inputs
    hidden_outputs = relu.(hidden_inputs)

    # Calculate the signal entering the output layer
    final_inputs = net.who * hidden_outputs
    final_outputs = relu.(final_inputs)

    # PART 2：Compare the resulting output with the desired output to guide the update of network weights
    # Output layer error = (target - actual)
    output_errors = targets - final_outputs
    hidden_errors = net.who' * output_errors

    net.who += net.lr .* (output_errors .* final_outputs .* (1.0 .- final_outputs)) * hidden_outputs'
    net.wih += net.lr .* (hidden_errors .* hidden_outputs .* (1.0 .- hidden_outputs)) * inputs'
end

# Start here
using Dates

start = Dates.now()
# parameters
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
epochs = 10
train_data_size = 6000
test_data_size = 1000

net_test = InitNet(input_nodes, hidden_nodes, output_nodes, learning_rate);
# import training set
# training_data_file = readdlm("mnist_train.csv", ',', use_mmap = true);
# import test set
# test_data_file = readdlm("mnist_test.csv", ',', use_mmap = true);
# data sets size
train_x, train_y = MNIST.traindata() # image data, image labels
test_x,  test_y  = MNIST.testdata() # image data, image labels


# Cycle training
for e = 1:epochs
# Training neural network
    for record = 1:train_data_size
        inputs = train_x[:,:,record]
        inputs = reshape(inputs, input_nodes, 1) # Adjustment dimension
        targets = zeros(output_nodes) .+ 0.01
        targets[round(Int, train_y[record]) + 1] = 0.99
        targets = reshape(targets, 10, 1)
        Train!(net_test, inputs, targets)
    end
end
# Effect test
scorecard = []

for record = 1:test_data_size
    correct_label = Int(test_y[record])
    inputs = test_x[:,:,record]
    inputs = reshape(inputs, input_nodes, 1) # Adjustment dimension
    outputs = Query(net_test, inputs)
    label = findmax(outputs)[2][1]
    if (label - 1 == correct_label)
        append!(scorecard, 1)
    else
        append!(scorecard, 0)
    end
end
# Accuracy
print("\nPerformance = ", sum(scorecard) / length(scorecard))

stop = Dates.now()

# Time taken
print("\n",0.001(stop-start).value, " seconds")
