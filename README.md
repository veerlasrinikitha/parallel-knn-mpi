# parallel-knn-mpi
This  implements a parallel k-Nearest Neighbors (k-NN) classifier using the Message Passing Interface (MPI) to efficiently classify large datasets across multiple processes. The implementation is in Python, leveraging mpi4py for parallelism, NumPy and scipy for numerical computations, and scikit-learn for evaluating classification accuracy
# Features
- Parallel Processing: Utilizes mpi4py to distribute test data and computation across multiple processes for faster k-NN classification.
- Flexible k Values: Supports evaluation for various k values (8, 16, 32, 64) in a single run.
- Performance Metrics: Reports accuracy, computation time, speedup, and efficiency for each configuration.
- Scalability Analysis: Prints strong and weak scalability insights as process count or data size changes.
# Install Packages from requirements.txt 
```bash
!pip install -r requirement.txt
```

