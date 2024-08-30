# Matrix Library Framework

## Overview

This Matrix Library Framework is a custom-designed C++ matrix manipulation library intended for mathematical operations commonly used in machine learning and neural networks. The framework provides a comprehensive set of matrix operations, including basic arithmetic, advanced transformations, and specialized functions for neural networks. It emphasizes efficiency and precision, leveraging unique pointer-based memory management and various matrix manipulation techniques essential for high-performance computing.

The library is designed to be easily integrated into machine learning projects, particularly those requiring custom implementations of neural network algorithms. With a focus on mathematical rigor, this framework provides the foundational tools necessary for implementing complex machine learning models from scratch.

## Features

- **Matrix Creation**: Create matrices with customizable dimensions and initial values.
- **Matrix Operations**: Perform matrix addition, subtraction, multiplication, transposition, and more.
- **Neural Network Operations**: Specialized functions for Hadamard products, broadcasting, and scalar operations.
- **Randomization Functions**: Initialize matrices using random values or specific distributions like Xavier initialization.
- **Input/Output Preprocessing**: Convert 2D array data into matrices suitable for machine learning models with stride and step adjustments.

## Matrix Class

### Matrix Constructors

- **`Matrix(int rows, int columns)`**: 
  Creates a matrix with specified rows and columns. Initializes all elements to zero.
  
- **`Matrix(int rows, int columns, float value)`**: 
  Creates a matrix with specified rows and columns, initializing all elements to the given value.
  
- **`Matrix(const Matrix& other)`**: 
  Copy constructor that creates a new matrix by copying an existing matrix. Ensures deep copy.
  
- **`Matrix(Matrix&& other) noexcept`**: 
  Move constructor that transfers ownership of data from another matrix. Useful for optimizing performance by avoiding deep copies.

### Member Functions

- **`Matrix& operator=(const Matrix& other)`**: 
  Assignment operator for copying data from one matrix to another. Ensures deep copy and checks for matching dimensions.

- **`Matrix& operator=(Matrix&& other) noexcept`**: 
  Move assignment operator for transferring ownership of data from another matrix. Useful for performance optimization.

- **`float& operator()(int row, int column)`**: 
  Element access operator (non-const) for modifying matrix elements at a given position.

- **`const float& operator()(int row, int column) const`**: 
  Element access operator (const) for reading matrix elements at a given position.

- **`int rows() const`**: 
  Returns the number of rows in the matrix.

- **`int columns() const`**: 
  Returns the number of columns in the matrix.

- **`static void Print(const Matrix& matrix)`**: 
  Prints the matrix to the console for easy visualization.

## Core Operations

### `void Matrix_Multiply(Matrix& result, const Matrix& first, const Matrix& second)`

Multiplies two matrices and stores the result in the provided matrix. The result matrix must have appropriate dimensions. This function checks for dimension compatibility and ensures that the result matrix is distinct from the input matrices.

### `Matrix Matrix_AutoCreate(const Matrix& first, const Matrix& second)`

Automatically creates a result matrix with the appropriate dimensions for matrix multiplication based on the input matrices.

### `void Matrix_Add(Matrix& result, const Matrix& matrix1, const Matrix& matrix2)`

Performs element-wise addition of two matrices and stores the result in the provided matrix. It ensures that all matrices involved have matching dimensions.

### `void Matrix_Subtract(Matrix& result, const Matrix& matrix1, const Matrix& matrix2)`

Performs element-wise subtraction of two matrices and stores the result in the provided matrix. It ensures that all matrices involved have matching dimensions.

### `void Matrix_Transpose(Matrix& final, const Matrix& original)`

Transposes the given matrix and stores the result in the final matrix. The result matrix must have appropriate dimensions to store the transposed data.

## Neural Network Operations

### `void Matrix_Hadamard_Product(Matrix& result, const Matrix& a, const Matrix& b)`

Computes the Hadamard (element-wise) product of two matrices and stores the result in the provided matrix. This function ensures that all matrices involved have matching dimensions.

### `void Matrix_Broadcast(Matrix& result, const Matrix& original, int newRows, int newColumns)`

Broadcasts an existing matrix to a larger size by repeating its elements according to the specified new dimensions. The result matrix must match the new dimensions.

### `void Matrix_Scalar_Multiply(Matrix& matrix, float scalar)`

Multiplies all elements of a matrix by a scalar value, modifying the matrix in place.

### `float Matrix_Sum_All_Elements(const Matrix& matrix)`

Calculates and returns the sum of all elements in the matrix.

### `void Matrix_Power(Matrix& matrix, float power)`

Raises each element of the matrix to the specified power, modifying the matrix in place.

### `void Matrix_Absolute(Matrix& result, const Matrix& original)`

Computes the absolute value of each element in the original matrix and stores the result in the provided matrix. The result matrix must have appropriate dimensions.

### `void Matrix_Sum_Columns(Matrix& dest, const Matrix& src)`

Sums the elements of each column in the source matrix and stores the result in the destination matrix. The destination matrix must have the same number of columns as the source matrix.

## Randomization Functions

### `void Matrix_Randomize(Matrix& matrix, float range = 3.0f)`

Randomizes the elements of the matrix using a uniform distribution within the specified range. The default range is from -3.0 to 3.0.

### `void Matrix_Xavier_Uniform(Matrix& matrix)`

Initializes the matrix using the Xavier uniform distribution, which is commonly used in neural network initialization to maintain a balanced variance across layers.

## Input/Output Preprocessing Functions

### `Matrix Matrix_Data_Preprocessor(int desiredRows, int desiredColumns, int stride, int step, const std::vector<std::vector<float>>& data)`

Converts 2D array data into a matrix, allowing for custom strides and steps to process the data effectively. This is particularly useful for preparing data for neural network training, where specific dimensions and overlaps are required.



## Getting Started

### Prerequisites

To build and use this Matrix Library, you'll need:

- A C++ compiler with C++23 support.
- CMake version 3.26 or later.

### Building the Library

1. **Clone the Repository**:
   First, clone the repository to your local machine

2. **Build with CMake**:
   You can build the Matrix Library as a shared library using the provided CMake configuration.

   ```sh
   mkdir build
   cd build
   cmake ..
   make
   ```

   This will generate the shared library `libMatrixLibrary.so` (or `MatrixLibrary.dll` on Windows) in the `build` directory.

### Using the Library in Your Project

To use the Matrix Library in your own project:

1. **Include the Header**:
   Make sure your project includes the `MatrixLibrary.h` header file. You can do this by adding the following line to your C++ source files:

   ```cpp
   #include "MatrixLibrary.h"
   ```

2. **Link the Library**:
   When compiling your project, link against the shared library generated in the build step. Depending on your build system, you can do this as follows:

   - **Using CMake**:
     If your project uses CMake, modify your `CMakeLists.txt` to include the Matrix Library:

     ```cmake
     # Find the MatrixLibrary package
     find_package(MatrixLibrary REQUIRED)

     # Include the headers
     include_directories(${MatrixLibrary_INCLUDE_DIRS})

     # Link the library
     target_link_libraries(YourTargetName ${MatrixLibrary_LIBRARIES})
     ```

   - **Using g++ or clang++**:
     If you're using a command-line tool like `g++` or `clang++`, you can link the library during compilation:

     ```sh
     g++ -o your_program your_source.cpp -L/path/to/library -lMatrixLibrary
     ```

3. **Run Your Program**:
   Ensure that the shared library is accessible at runtime, either by placing it in the same directory as your executable or by setting the appropriate environment variable (`LD_LIBRARY_PATH` on Linux, `PATH` on Windows).

