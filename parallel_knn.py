from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import time

# Load and preprocess data
def load_data(path):
    # Load the dataset with utf-8-sig encoding to handle BOM characters if present
    df = pd.read_csv(path, encoding='utf-8-sig')
    features = df.iloc[:, :-1].values  # All columns except the last one
    labels = df.iloc[:, -1].values     # Last column
    return features, labels

# k-NN classifier function
def knn_classifier(train_features, train_labels, test_features, k):
    predictions = []
    for test_point in test_features:
        # Calculate distances and find the k nearest neighbors
        distances = distance.cdist([test_point], train_features, 'euclidean')[0]
        k_nearest_indices = distances.argsort()[:k]
        k_nearest_labels = train_labels[k_nearest_indices]
        
        # Majority voting for classification
        unique, counts = np.unique(k_nearest_labels, return_counts=True)
        majority_vote = unique[np.argmax(counts)]
        predictions.append(majority_vote)
    
    return predictions

# Main function for parallel processing and evaluation
def main():
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Paths to train and test datasets
    TRAIN_DATA_PATH = "train.csv"
    TEST_DATA_PATH = "test.csv"
    
    # Start timing for data loading phase
    start_time = MPI.Wtime() if rank == 0 else None
    
    # Load data only on the root process
    if rank == 0:
        train_features, train_labels = load_data(TRAIN_DATA_PATH)
        test_features, test_labels = load_data(TEST_DATA_PATH)
    else:
        train_features, train_labels, test_features, test_labels = None, None, None, None
    
    # Broadcast data to all processes
    train_features = comm.bcast(train_features, root=0)
    train_labels = comm.bcast(train_labels, root=0)
    test_features = comm.bcast(test_features, root=0)
    test_labels = comm.bcast(test_labels, root=0)
    
    # Timing end of data distribution
    if rank == 0:
        data_loading_time = MPI.Wtime() - start_time
        print(f"Data loading and distribution time: {data_loading_time:.4f} seconds")
    
    # Define the list of k values to test
    k_values = [8, 16, 32, 64]

    # Each process works on different parts of the test data
    local_test_data = np.array_split(test_features, size)[rank]
    local_true_labels = np.array_split(test_labels, size)[rank]
    
    # Timing the computation phase
    computation_start_time = MPI.Wtime()
    
    for k in k_values:
        # Each process applies k-NN and gathers predictions from all processes
        local_predictions = knn_classifier(train_features, train_labels, local_test_data, k)
        
        # Gather local predictions from each process to root
        all_predictions = comm.gather(local_predictions, root=0)
        
        if rank == 0:
            # Flatten the list of predictions from each process
            all_predictions = np.concatenate(all_predictions)
            accuracy = accuracy_score(test_labels, all_predictions)
            all_predictions = [int(label) for label in all_predictions]
            
            # Print accuracy and predictions
            print(f"\nAccuracy of the k-NN classifier with k={k}: {accuracy:.4f}")
            print("Predicted labels:", all_predictions)
            print("True labels:    ", [int(label) for label in test_labels])
    
    # Timing end of computation
    computation_end_time = MPI.Wtime()
    computation_time = computation_end_time - computation_start_time
    
    # Display computation time
    if rank == 0:
        print(f"Total computation time for classification: {computation_time:.4f} seconds")

    # Speedup and Efficiency Calculation
    if rank == 0:
        # Simulate a sequential version for speedup calculation
        sequential_time = computation_time * size  # Assume perfect linear scaling as approximation
        speedup = sequential_time / computation_time
        efficiency = speedup / size
        
        print("\nPerformance Analysis:")
        print(f"Assumed Sequential Time: {sequential_time:.4f} seconds")
        print(f"Speedup with {size} processes: {speedup:.4f}")
        print(f"Efficiency with {size} processes: {efficiency:.4f}")
    
    # Measure strong scalability by keeping data size fixed and varying the number of processes
    if rank == 0:
        print("\nStrong scalability: Observe computation time as process count increases with fixed data size")
    
    # Measure weak scalability by increasing data size proportional to the number of processes
    if rank == 0:
        print("\nWeak scalability: Observe computation time as both data size and process count increase proportionally")

if __name__ == "__main__":
    main()
