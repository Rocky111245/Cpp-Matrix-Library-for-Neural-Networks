#include "library.h"
#include <iostream>
#include <memory>
#include <algorithm> // for std::fill
#include <stdexcept> // for std::invalid_argument
#include <random>
#include <vector>


class Matrix {
public:

    // Primary Constructor: Initialize all elements to zero
    Matrix(int rows, int columns)
            : rows_(rows), columns_(columns), data_(std::make_unique<float[]>(rows * columns)) {
        if (rows <= 0 || columns <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive integers.");
        }
        // Initialize all elements to zero
        std::fill(data_.get(), data_.get() + rows * columns, 0.0f);
    }

    // Secondary Constructor: Initialize all elements to a specified value
    Matrix(int rows, int columns, float value)
            : rows_(rows), columns_(columns), data_(std::make_unique<float[]>(rows * columns)) {
        if (rows <= 0 || columns <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive integers.");
        }
        // Initialize all elements to the specified value
        std::fill(data_.get(), data_.get() + rows * columns, value);
    }




    // Copy constructor for deep copying
    Matrix(const Matrix& other)
            : rows_(other.rows_), columns_(other.columns_), data_(std::make_unique<float[]>(other.rows_ * other.columns_)) {
        std::copy(other.data_.get(), other.data_.get() + (other.rows_ * other.columns_), data_.get());
        std::cout << "Matrix copied\n";
    }

    // Move constructor
    Matrix(Matrix&& other) noexcept
            : rows_(other.rows_), columns_(other.columns_), data_(std::move(other.data_)) {
        other.rows_ = 0;
        other.columns_ = 0;
        std::cout << "Matrix moved\n";
    }




    //Helper functions







    // Static print function
    static void Print(const Matrix& matrix) {
        for (int i = 0; i < matrix.rows_; ++i) {
            for (int j = 0; j < matrix.columns_; ++j) {
                std::cout << matrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    // Access and modify element (non-const version)
    float& operator()(int row, int column) {
        return data_[row * columns_ + column];
    }

    // Access element (const version)
    const float& operator()(int row, int column) const {
        return data_[row * columns_ + column];
    }


    // Get number of rows
    int rows() const {
        return rows_;
    }

    // Get number of columns
    int columns() const {
        return columns_;
    }


private:
    int rows_;
    int columns_;
    std::unique_ptr<float[]> data_;

};



//// Core Operations


// Matrix Multiplication Function
void Matrix_Multiply( const Matrix& first,const Matrix& second,Matrix& result) {

    if (&result == &first || &result == &second) {
        throw std::invalid_argument("Result matrix must be different from input matrices.");
    }

    if (first.columns() != second.rows()) {
        throw std::invalid_argument("Number of columns in the first matrix must equal the number of rows in the second matrix.");
    }

    if (result.rows() != first.rows() || result.columns() != second.columns()) {
        throw std::invalid_argument("Result matrix dimensions do not match the dimensions required for multiplication.");
    }

    for (int i = 0; i < first.rows(); ++i) {
        for (int j = 0; j < second.columns(); ++j) {
            float sum = 0.0f;
            for (int k = 0; k < first.columns(); ++k) {
                sum += first(i, k) * second(k, j);
            }
            result(i, j) = sum;
        }
    }
}

//This is a function that automatically creates a Matrix object based on desired Matrices that
//will be used during multiplication

// Helper function to create a result matrix for multiplication
Matrix Matrix_Autocreate(const Matrix& first, const Matrix& second) {
    if (first.columns() != second.rows()) {
        throw std::invalid_argument("Number of columns in the first matrix must equal the number of rows in the second matrix.");
    }
    return Matrix(first.rows(), second.columns());
}



// Matrix Addition Function
void Matrix_Add(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {

    if (&result == &matrix1 || &result == &matrix2) {
        throw std::invalid_argument("Result matrix must be different from input matrices.");
    }
    if (matrix1.rows() != matrix2.rows() || matrix1.columns() != matrix2.columns()) {
        throw std::invalid_argument("Matrices dimensions do not match.");
    }

    if (result.rows() != matrix1.rows() || result.columns() != matrix1.columns()) {
        throw std::invalid_argument("Result matrix dimensions do not match the dimensions required for addition.");
    }

    for (int i = 0; i < matrix1.rows(); ++i) {
        for (int j = 0; j < matrix1.columns(); ++j) {
            result(i, j) = matrix1(i, j) + matrix2(i, j);
        }
    }
}

// Matrix subtraction function
void Matrix_Subtract(const Matrix& matrix1, const Matrix& matrix2,Matrix& result) {
    if (&result == &matrix1 || &result == &matrix2) {
        throw std::invalid_argument("Result matrix must be different from input matrices.");
    }

    if (matrix1.rows() != matrix2.rows() || matrix1.columns() != matrix2.columns()) {
        throw std::invalid_argument("Matrices dimensions do not match.");
    }

    if (result.rows() != matrix1.rows() || result.columns() != matrix1.columns()) {
        throw std::invalid_argument("Result matrix dimensions do not match the dimensions required for subtraction.");
    }

    for (int i = 0; i < matrix1.rows(); ++i) {
        for (int j = 0; j < matrix1.columns(); ++j) {
            result(i, j) = matrix1(i, j) - matrix2(i, j);
        }
    }
}

// Function to transpose a matrix
void Matrix_Transpose(Matrix& final, const Matrix& original) {
    if (final.rows() != original.columns() || final.columns() != original.rows()) {
        throw std::invalid_argument("Final matrix dimensions do not match the transposed dimensions of the original matrix.");
    }

    for (int i = 0; i < original.rows(); ++i) {
        for (int j = 0; j < original.columns(); ++j) {
            final(j, i) = original(i, j);
        }
    }
}


//// Operations needed for Neural Networks

// Function to perform Hadamard Product (element-wise multiplication)
void Matrix_Hadamard_Product(Matrix& result, const Matrix& a, const Matrix& b) {
    if (&result == &a || &result == &b) {
        throw std::invalid_argument("Result matrix must be different from input matrices.");
    }
    if (a.rows() != b.rows() || a.columns() != b.columns()) {
        throw std::invalid_argument("Matrices dimensions do not match.");
    }

    if (result.rows() != a.rows() || result.columns() != a.columns()) {
        throw std::invalid_argument("Result matrix dimensions do not match the dimensions required for Hadamard product.");
    }

    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.columns(); ++j) {
            result(i, j) = a(i, j) * b(i, j);
        }
    }
}

// Function to broadcast an existing matrix to a larger size specified by newRows and newColumns
void Matrix_Broadcast(Matrix& result, const Matrix& original, int newRows, int newColumns) {
    if (newRows % original.rows() != 0 || newColumns % original.columns() != 0) {
        throw std::invalid_argument("New dimensions must be multiples of original dimensions.");
    }

    if (result.rows() != newRows || result.columns() != newColumns) {
        throw std::invalid_argument("Result matrix dimensions do not match the specified new dimensions.");
    }

    for (int i = 0; i < newRows; ++i) {
        for (int j = 0; j < newColumns; ++j) {
            int originalRow = i % original.rows();
            int originalColumn = j % original.columns();
            result(i, j) = original(originalRow, originalColumn);
        }
    }
}


// Function to multiply all elements of a matrix by a scalar value
void Matrix_Scalar_Multiply(Matrix& matrix, float scalar) {
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.columns(); ++j) {
            matrix(i, j) *= scalar;
        }
    }
}

// Function to add up all values in a matrix and return a single scalar
float Matrix_Sum_All_Elements(const Matrix& matrix) {
    float totalSum = 0;
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.columns(); ++j) {
            totalSum += matrix(i, j);
        }
    }
    return totalSum;
}

// Function to raise each element of a matrix to a specified power
void Matrix_Power(Matrix& matrix, float power) {
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.columns(); ++j) {
            matrix(i, j) = std::pow(matrix(i, j), power);
        }
    }
}

// Function to compute the absolute values of a matrix
void Matrix_Absolute(Matrix& result, const Matrix& original) {
    if (&result == &original) {
        throw std::invalid_argument("Result matrix must be different from input matrices.");
    }
    if (result.rows() != original.rows() || result.columns() != original.columns()) {
        throw std::invalid_argument("Result matrix dimensions must match the original matrix dimensions.");
    }

    for (int i = 0; i < original.rows(); ++i) {
        for (int j = 0; j < original.columns(); ++j) {
            result(i, j) = std::abs(original(i, j));
        }
    }
}

// Function to sum columns of a matrix and store in the destination matrix
void Matrix_SumColumns(Matrix& dest, const Matrix& src) {

    if (&dest == &src) {
        throw std::invalid_argument("Result matrix must be different from input matrices.");
    }

    if (dest.columns() != src.columns()) {
        throw std::invalid_argument("Destination matrix must have the same number of columns as the source matrix.");
    }

    for (int col = 0; col < src.columns(); ++col) {
        float column_sum = 0;
        for (int row = 0; row < src.rows(); ++row) {
            column_sum += src(row, col);
        }
        for (int row = 0; row < dest.rows(); ++row) {
            dest(row, col) = column_sum;
        }
    }
}














//// Randomization Functions

// Function to randomize an existing matrix
void Matrix_Randomize(Matrix& matrix,float range) {
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<float> dist(-range, range); // Uniform distribution in range [-1, 1]

    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.columns(); ++j) {
            matrix(i, j) = dist(gen);
        }
    }
}

// Function to initialize a matrix with Xavier uniform distribution
void Matrix_Xavier_Uniform(Matrix& matrix) {
    int rows = matrix.rows();
    int columns = matrix.columns();
    float limit = std::sqrt(6.0f / (rows + columns));  // Xavier initialization limit

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            matrix(i, j) = dist(gen);
        }
    }
}



//// Input Output Preprocessing Functions



Matrix CreateMatrixWithStride(int maxColumns, int totalRows, int desiredRows, int desiredColumns, int stride, int step, const std::vector<float>& data) {
    Matrix result(desiredRows, desiredColumns);

    int index = 0;
    for (int i = 0; i < desiredRows && (i + step) < totalRows; ++i) {
        int baseRow = i + step;
        for (int j = stride; j < desiredColumns + stride; ++j) {
            if (j < maxColumns) {
                int dataIndex = baseRow * maxColumns + j;
                result(i, j - stride) = data[dataIndex];
            } else {
                result(i, j - stride) = 0;
            }
        }
    }
    return result;
}













int main(){
    Matrix A(3,2,2);
    Matrix B(3,2,3);
    Matrix::Print(A);
    Matrix_Add(A,A,B);
    Matrix::Print(A);
    return 0;



}
