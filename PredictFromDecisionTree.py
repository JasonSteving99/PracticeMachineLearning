from DecisionTreeNode import DecisionTreeNode

def predict(trained_decision_tree, sample):
    test_feature_index = trained_decision_tree.test_feature_index
    if test_feature_index == None:
        return trained_decision_tree.majority_vote
    elif trained_decision_tree.children[sample[test_feature_index]] == None:
        return trained_decision_tree.majority_vote
    else:
        return predict(trained_decision_tree.children[sample[test_feature_index]], sample)
