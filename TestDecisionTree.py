import matplotlib.pyplot as plt
import numpy as np

import LearnDecisionTree as L
import PredictFromDecisionTree as P
import ProcessZooTrainingData as Z


'''
Returns a 2D int-array of uniformly generated samples with each row representing a sample.

num_samples: the number of rows in the returned array.
feature_dimensions: a list of ints representing the total number of values that each feature can take on.
                    the returned samples will uniformly choose values in this range.
'''
def genSamples(num_samples, dimensions):
    num_features = len(dimensions)
    samples = np.array([0] * num_samples * num_features).reshape(num_samples, num_features)
    for d, dim_size in enumerate(dimensions):
        samples[:,d] = np.random.choice(a = dim_size, size = num_samples)
    return samples

def testPredictionAccuracy(trained_decision_tree, test_sample_set, goal_feature_index, verbosity = 'QUIET'):
    num_correct = 0
    for test_sample in test_sample_set:
        sample_prediction = P.predict(trained_decision_tree, test_sample)
        if verbosity == 'VERBOSE':
            print('Prediction:{0} - Sample Label:{1}'.format(sample_prediction, test_sample[goal_feature_index]))
        if sample_prediction == test_sample[goal_feature_index]:
            num_correct += 1
    percent_accuracy = num_correct/len(test_sample_set)
    if verbosity != 'SILENT':
        print(percent_accuracy)
    return percent_accuracy


feature_dimensions = [2,3,2,4,2,2,3,2]
goal_feature_index = len(feature_dimensions) - 1
generated_samples = genSamples(100, feature_dimensions)
num_training_samples = 80
training_samples = generated_samples[:num_training_samples]
test_samples = generated_samples[num_training_samples:]
tested_features = [False] * len(feature_dimensions)
# TODO(steving) THIS IS A MAAAAJOOOR HACK. FIX THE LEARNING CODE INSTEAD OF THIS, BECAUSE THIS IS BASICALLY MASKING A BUG.
tested_features[goal_feature_index] = True
trained_decision_tree = L.LearnDecisionTree(training_samples, feature_dimensions, goal_feature_index, tested_features)

# Test Predictive power on the new test samples.
# testPredictionAccuracy(trained_decision_tree, genSamples(200, feature_dimensions), goal_feature_index, verbosity='QUIET')
testPredictionAccuracy(trained_decision_tree, test_samples, goal_feature_index, verbosity='QUIET')
# Should get very high accuracy on the training samples every time.
print('Test Accuracy by passing the training samples back through:')
testPredictionAccuracy(trained_decision_tree, training_samples, goal_feature_index = 7, verbosity='QUIET')

random_sample = genSamples(1, feature_dimensions)[0]
print(random_sample)
print('Prediction: {0}'.format(P.predict(trained_decision_tree, random_sample)))



# BELOW HERE TEST THE DECISION TREE WITH ZOO DATA FOUND ONLINE INSTEAD OF RANDOMLY GENERATED DATA:
zoo_samples = Z.parseZooData()
zoo_feature_dimensions = Z.ZOO_FEATURE_DIMENSIONS
zoo_goal_feature_index = len(zoo_feature_dimensions) - 1
zoo_tested_features = [False] * len(zoo_feature_dimensions)
# TODO(steving) THIS IS A MAAAAJOOOR HACK. FIX THE LEARNING CODE INSTEAD OF THIS, BECAUSE THIS IS BASICALLY MASKING A BUG.
zoo_tested_features[zoo_goal_feature_index] = True

raw_test_results = [[0 for _ in range(20)] for _ in range(100)]
averaged_test_results = [0] * 100
# Divide the zoo data into a training set and a testing set.
for num_zoo_training_samples in range(1, 100):
    # Average 20 tests on random splits at each training sample subset size to get better data.
    for test in range(20):
        sample_indices_filter = lambda zoo_sample_indices : np.array(list(map(lambda i: zoo_samples[i], zoo_sample_indices)))
        training_sample_indices = np.random.choice(len(zoo_samples), num_zoo_training_samples, replace=False)
        zoo_training_samples = sample_indices_filter(training_sample_indices)
        test_sample_indices = set(range(len(zoo_samples))).difference(training_sample_indices)
        zoo_test_samples = sample_indices_filter(test_sample_indices)
        # zoo_training_samples = zoo_samples[:num_zoo_training_samples]
        # zoo_test_samples = zoo_samples[num_zoo_training_samples:]

        # Train a new decision tree for the zoo data.
        trained_zoo_data_decision_tree = L.LearnDecisionTree(zoo_training_samples, zoo_feature_dimensions, zoo_goal_feature_index, zoo_tested_features)

        # Test Predictive power on the test samples.
        percent_accuracy = testPredictionAccuracy(trained_zoo_data_decision_tree, zoo_test_samples, zoo_goal_feature_index, verbosity='SILENT')
        # We'll incrementally build the sum for the average.
        raw_test_results[num_zoo_training_samples][test] = percent_accuracy
        averaged_test_results[num_zoo_training_samples] += percent_accuracy / 20


plt.xlabel("Number of Training Samples")
plt.ylabel('Accuracy')
average_results_line, = plt.plot(list(range(100)), averaged_test_results, 'b-')
for num_zoo_training_samples in range(1, 100):
    plt.scatter([num_zoo_training_samples]*20, raw_test_results[num_zoo_training_samples])
average_results_line.set_label('Test Results Average')
plt.legend(loc='lower right')
plt.show()