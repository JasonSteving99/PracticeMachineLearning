class DecisionTreeNode:
    def __init__(self, feature_dimensions, test_feature_index, majority_vote):
        self.test_feature_index = test_feature_index
        if self.test_feature_index == None:
            self.children = None
        else:
            self.children = [None] * feature_dimensions[test_feature_index]
        self.majority_vote = majority_vote

    
    def __str__(self):
        return 'Test Feature Index: {0}\nMajority Vote: {1}\n'.format(self.test_feature_index, self.majority_vote)