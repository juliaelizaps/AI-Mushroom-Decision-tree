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

info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)

info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)

# UNIT TESTS
compute_information_gain_test(compute_information_gain)

# UNQ_C4
# GRADED FUNCTION: get_best_split

def get_best_split(X, y, node_indices):   
    num_features = X.shape[1]
    best_feature = -1
    max_info_gain = 0
    for feature in range(num_features): 
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:  
            max_info_gain = info_gain
            best_feature = feature        
    return best_feature

best_feature = get_best_split(X_train, y_train, root_indices)
print("Best feature to split on: %d" % best_feature)
# UNIT TESTS
get_best_split_test(get_best_split)

# Not graded
tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):

    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
  
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))

    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    

    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)

build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
generate_tree_viz(root_indices, y_train, tree)