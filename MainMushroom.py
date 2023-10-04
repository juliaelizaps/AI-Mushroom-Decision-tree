import numpy as np
import matplotlib.pyplot as plt
from public_tests import *
from utils import *

%matplotlib inline

X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

print("First few elements of X_train:\n", X_train[:5])
print("Type of X_train:",type(X_train))

print("First few elements of y_train:", y_train[:5])
print("Type of y_train:",type(y_train))

print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))

def compute_entropy(y):

    entropy = 0.
    
    p1 = np.mean(y)
    
    if p1 == 0 or p1 == 1:
        return 0
    entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)               
    
    return entropy

print("Entropy at root node: ", compute_entropy(y_train)) 

# UNIT TESTS
compute_entropy_test(compute_entropy)


def split_dataset(X, node_indices, feature):
    left_indices = []
    right_indices = []
    for i in node_indices:   
        if X[i, feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i) 
    return left_indices, right_indices

# Case 1
root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
feature = 0
left_indices, right_indices = split_dataset(X_train, root_indices, feature)
print("CASE 1:")
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)
generate_split_viz(root_indices, left_indices, right_indices, feature)
print()

# Case 2
root_indices_subset = [0, 2, 4, 6, 8]
left_indices, right_indices = split_dataset(X_train, root_indices_subset, feature)
print("CASE 2:")
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)
generate_split_viz(root_indices_subset, left_indices, right_indices, feature)

# UNIT TESTS    
split_dataset_test(split_dataset)

# UNQ_C3
# GRADED FUNCTION: compute_information_gain
def compute_information_gain(X, y, node_indices, feature):

    left_indices, right_indices = split_dataset(X, node_indices, feature)
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    information_gain = 0
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    if len(y_left) == 0 or len(y_right) == 0:
        information_gain = 0.0
    else:
        w_left = len(left_indices) / len(node_indices)
        w_right = len(right_indices) / len(node_indices)
        weighted_entropy = w_left * left_entropy + w_right * right_entropy
        information_gain = node_entropy - weighted_entropy
    return information_gain

