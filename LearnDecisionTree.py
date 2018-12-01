from math import log

from DecisionTreeNode import DecisionTreeNode
import numpy as np


'''
TODO(steving) CURRENTLY THIS IS ACTUALLY BROKEN IN THAT IT WILL SPLIT ON THE GOAL FEATURE INDEX SINCE YOU HAVEN'T 
PREVENTED IT FROM DOING THAT!! YOU CAN HACK THIS TO WORK BY SETTING THE GOAL FEATURE INDEX TO True IN THE
tested_features ARGUMENT TO PREVENT THIS IMPLEMENTATION FROM SPLITTING ON THAT FEATURE. FIX THIS AT SOME POINT.
'''
def LearnDecisionTree(training_sample_set, feature_dimensions, goal_feature_index, tested_features):
    # We've trained on no examples that have this particular combination of features that got us to this point, so return None.
    # When traversing the decision tree to make a prediction for a new sample, we'll have no better option than to take the 
    # Majority vote of the node which contains this None child.
    if len(training_sample_set) == 0:
        return None
    curr_majority_vote, curr_majority_vote_magnitude = getMajorityVote(training_sample_set, feature_dimensions, goal_feature_index)
    # If there're no more features to split on, then the best we can do is return the majority vote here.
    if len(list(filter(lambda b:b, tested_features))) == len(tested_features):
        return DecisionTreeNode(feature_dimensions, None, curr_majority_vote)
    # There is 100% consensus (0 entropy) in the given training sample set, so if this node in the decision tree is reached
    # while trying to make a prediction for a new sample, the current unanimous vote for the goal feature value should be chosen.
    elif curr_majority_vote_magnitude == len(training_sample_set):
        return DecisionTreeNode(feature_dimensions, None, curr_majority_vote)
    test_feature_index, test_feature_splits = determineBestTestFeature(training_sample_set, feature_dimensions, goal_feature_index, tested_features)
    updated_tested_features = tested_features.copy()
    updated_tested_features[test_feature_index] = True
    curr_node = DecisionTreeNode(feature_dimensions, test_feature_index, curr_majority_vote)
    curr_node.children = list(map(lambda split: LearnDecisionTree(split, feature_dimensions, goal_feature_index, updated_tested_features), test_feature_splits))
    return curr_node
    

def determineBestTestFeature(sample_set, feature_dimensions, goal_feature_index, tested_features):
    remaining_features_to_test = list(map(lambda p: p[0], filter(lambda p: not p[1], enumerate(tested_features))))
    prior_entropy = H(sample_set, feature_dimensions, goal_feature_index)
    #Information gain is strictly positive, so we could similarly just do comparisons for the min expected remainder entropy.
    max_information_gain = -1
    best_test_feature_index = -1
    best_test_feature_splits = []
    for test_feature in remaining_features_to_test:
        curr_test_feature_splits = splitSampleSet(sample_set, feature_dimensions, test_feature)
        curr_test_feature_information_gain = prior_entropy - expectedRemainderEntropy(curr_test_feature_splits, feature_dimensions, goal_feature_index)
        if curr_test_feature_information_gain > max_information_gain:
            max_information_gain = curr_test_feature_information_gain
            best_test_feature_index = test_feature
            best_test_feature_splits = curr_test_feature_splits
    return best_test_feature_index, best_test_feature_splits


'''
Returns the feature value with the largest number of votes given the splits over that particular feature.

sample_set: the set of samples to compute Entropy over (given as a numpy array).
'''
def getMajorityVote(sample_set, feature_dimensions, goal_feature_index):
    goal_feature_votes = splitSampleSet(sample_set, feature_dimensions, goal_feature_index)
    return max(enumerate(map(len, goal_feature_votes)), key = lambda p:p[1])

'''
Splits the given sample set into disjoint subsets grouped by common values for the given split feature.

sample_set: the set of samples to compute Entropy over (given as a numpy array).
feature_dimensions: a list of ints representing the total number of values that each feature can take on.
split_feature_index: the index of the feature to split on.
'''
def splitSampleSet(sample_set, feature_dimensions, split_feature_index):
    splits = [[] for _ in range(feature_dimensions[split_feature_index])]
    for sample in sample_set:
        splits[sample[split_feature_index]].append(sample)
    return list(map(np.array, splits))


'''
Computes and returns the expected entropy remaining with respect to the given goal feature after splitting
on a certain split feature. This function assumes that the given splits are already correctly partitioned
on a split feature as desired.

splits: the sets of samples for expected entropy to be computed over (given as a collection of numpy arrays).
feature_dimensions: a list of ints representing the total number of values that each feature can take on.
goal_feature_index: the index of the goal feature.
'''
def expectedRemainderEntropy(splits, feature_dimensions, goal_feature_index):
    split_entropy_list = np.array(list(map(lambda split: H(split, feature_dimensions, goal_feature_index), filter(lambda split: len(split) > 0, splits))))
    split_size_list = np.array(list(filter(lambda list_len: list_len > 0, map(len, splits))))
    num_samples = sum(split_size_list)
    return sum(split_entropy_list*split_size_list/num_samples)


'''
Computes and returns the Entropy of a set of samples with respect to the given goal feature.

sample_set: the set of samples to compute Entropy over (given as a numpy array).
feature_dimensions: a list of ints representing the total number of values that each feature can take on.
goal_feature_index: the index of the goal feature.
'''
def H(sample_set, feature_dimensions, goal_feature_index):
    # We'll count the total number of each observed value of the goal feature.
    goal_feature_value_counts = np.array([0] * feature_dimensions[goal_feature_index])
    for goal_feature_value in sample_set[:,goal_feature_index]:
        goal_feature_value_counts[goal_feature_value] += 1
    feature_value_entropy_term = lambda p : 0 if p == 0 else p * log(1./p, 2)
    return sum(map(feature_value_entropy_term, goal_feature_value_counts/len(sample_set)))
