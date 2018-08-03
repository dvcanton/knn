import numpy as np
from sklearn import datasets


class KNN:
    def __init__(self, training_set):
        self.trainingSet = training_set

    # Calculates the similarities between two instaces
    def distance(self, first_instance, second_instance):
        first_instance = np.array(first_instance)
        second_instance = np.array(second_instance)

        return np.linalg.norm(first_instance - second_instance)


    def getNeighbours(self, instance, k):
        # Check if trainingSet is not null or empty
        neighbours = []

        for element in training_set:
            distance = self.distance(instance, element)
            neighbours.append((element, distance))

        neighbours.sort(key=lambda x: x[1])
        return neighbours[1:k+1]


# Example code
data = datasets.load_iris().data

n_samples = 5
k = 1
indices = np.random.permutation(len(data))[:n_samples]
training_set = data[indices]

print(training_set)

knn = KNN(training_set)
instance = training_set[1]
neighbours = knn.getNeighbours(instance, k)

print(neighbours)
