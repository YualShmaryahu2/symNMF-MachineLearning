# symNMF-MachineLearning
This project encompasses the deployment of a clustering algorithm rooted in symmetric Non-negative Matrix Factorization (symNMF). Moreover, it includes an evaluation on various datasets, accompanied by a comparative analysis with the K-means algorithm.
-------------------
Symmetric Non-negative Matrix Factorization (symNMF) is a mathematical and computational technique used in the field of machine learning and data analysis. It is a variant of Non-negative Matrix Factorization (NMF) specifically designed for symmetric data, such as correlation matrices or data with inherent symmetrical patterns.
In symNMF, the goal is to factorize a given symmetric matrix into two lower-dimensional matrices, typically referred to as the basis and coefficient matrices. These matrices are constrained to contain only non-negative values. The factorization helps uncover underlying patterns and structures within the data, which is particularly useful in tasks like dimensionality reduction, feature extraction, and clustering.
Reference to SymNMF: 
Da Kuang, Chris Ding, and Haesun Park. Symmetric nonnegative matrix factorization for
graph clustering. In Proceedings of the 2012 SIAM International Conference on Data Mining
(SDM), Proceedings, pages 106–117. Society for Industrial and Applied Mathematics, April
2012.

In this project there are the following files:

symnmf.py:
Python interface of the code with the following requirements (user CMD arguments):
(a) k (int, < N): Number of required clusters.
(b) goal: Can get the following values:
i. symnmf: Perform full the symNMF.
ii. sym: Calculate and output the similarity matrix.
iii. ddg: Calculate and output the Diagonal Degree Matrix.
iv. norm: Calculate and output the normalized similarity matrix.
(c) file_name (.txt): The path to the Input file, it will contain N data points for all above
goals, the file extension is .txt
**This program outputs the required matrix.**

symnmf.c:
C interface of the code.
This is the C implementation program, with the following requirements (user CMD arguments):
(a) goal: Can get the following values:
i. sym: Calculate and output the similarity matrix.
ii. ddg: Calculate and output the Diagonal Degree Matrix.
iii. norm: Calculate and output the normalized similarity matrix.
(b) file_name (.txt ): The path to the Input file, it will contain N data points for all
above goals, the file extension is .txt
**This program outputs the required matrix.**

symnmfmodule.c:
Python C API wrapper.
In this file the C extension is defined which serves the functions: symnmf,sym,ddg,norm for Python.

symnmf.h:
C header file.
This header defines all functions prototypes that is being used in symnmfmodule.c and implemented at symnmf.c.

analysis.py:
Analyzing the algorithm.
Comparing SymNMF to Kmeans Algorithm. Applying both methods to given dataset and reporting the silhouette_score from the sklearn.metrics.

setup.py:
The setup file.
This is the build used to create the *.so file that will allow symnmf.py to import symnmfmodule.

Makefile:
The script for building symnmf executable, considering all it’s dependency. The compilation command should include all the flags as below:
gcc -ansi -Wall -Wextra -Werror -pedantic-errors
