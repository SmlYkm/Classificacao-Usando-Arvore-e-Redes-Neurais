package ml;

import java.util.stream.IntStream;
import ml.functions.MathFunction;
import java.util.Arrays;

public class Matrix {
    private final int     rows;
    private final int     cols;
    private final float[] arr;

    private static class MatrixException extends RuntimeException {
        MatrixException(int i, int j, Matrix mat) {
            super(
                String.format(
                    "Matrix index error: index out of range! Arguments are row = %d, col = %d, expected row in range [0, %d] and col in range[0, %d]", 
                    i, j, mat.rows, mat.cols
                )
            );
        }

        MatrixException(Matrix a, Matrix b, Matrix c) {
            super(abcErrorMsg(a, b, c));
        }

        MatrixException(Matrix a, Matrix b) {
            super(abErrorMsg(a, b));
        }

        private static String abErrorMsg(Matrix a, Matrix b) {
            String message = String.format("\nMatrix operation error: Dimension error!");
            if (a.rows != b.rows) 
                message += String.format("\n* rows\n\tExpected (a.rows = %d, b.rows = %d) but a.rows is %d and b.rows is %d!", a.rows, b.rows, a.rows, b.rows);
            if (a.cols != b.cols)
                message += String.format("\n* cols\n\tExpected (a.cols = %d, b.cols = %d) but a.cols is %d and b.cols is %d!", a.cols, b.cols, a.cols, b.cols);
            return message;
        }

        private static String abcErrorMsg(Matrix a, Matrix b, Matrix c) {
            String message = "Matrix multiplication error: Dimension error!";
            if (a.cols != b.rows)
                message += String.format("\n* a, b\n\tExpected (a.cols = %d, b.rows = %d) or (a.cols = %d, b.rows = %d) but a.cols is %d and b.rows is %d!", a.cols, a.cols, b.rows, b.rows, a.cols, b.rows);
            if (a.rows != c.rows)
                message += String.format("\n* a, this\n\tExpected (a.rows = %d, this.rows = %d) or (a.rows = %d, this.rows = %d) but a.rows is %d and this.rows is %d!", a.rows, a.rows, c.rows, c.rows, a.rows, c.rows);
            if (b.cols != c.cols)
                message += String.format("\n* b, this\n\tExpected (b.cols = %d, this.cols = %d) or (b.cols = %d, this.cols = %d) but b.cols is %d and this.cols is %d!", b.cols, b.cols, c.cols, c.cols, b.cols, c.cols);
            return message;
        }
    }
    
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.arr  = new float[rows * cols];
    }

    public Matrix(Matrix other) {
        rows = other.rows;
        cols = other.cols;
        arr  = new float[rows*cols];

        for (int i = 0; i < rows*cols; ++i)
            arr[i] = other.arr[i];
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public void copy(Matrix other) {
        int len = rows*cols;
        if (len != other.rows*other.cols)
            throw new MatrixException(this, other);
        
        for (int i = 0; i < rows; ++i)
            arr[i] = other.arr[i];
    }

    public float get(int i, int j) {
        if (i < 0 || i >= rows || j < 0 || j >= cols)
            throw new MatrixException(i, j, this);

        return arr[i*cols + j];
    }

    public void set(int i, int j, float value) {
        if (i < 0 || i >= rows || j < 0 || j >= cols)
            throw new MatrixException(i, j, this);

        arr[i*cols + j] = value;
    }

    public void zeroOut() {
        Arrays.fill(this.arr, 0.0f);
    }

    public void randomize() {
        for (int i = 0; i < rows*cols; ++i)
            arr[i] = (float) (Math.random() * 2.0 - 1.0);
    }

    // this = a * b
    public void mul(Matrix a, Matrix b) {
        if (a.cols != b.rows || a.rows != rows || b.cols != cols)
            throw new MatrixException(a, b, this);
        
        IntStream.range(0, rows).parallel().forEach((i) -> {
            for (int j = 0; j < cols; ++j) {
                float temp = 0;
                for (int k = 0; k < a.cols; ++k)
                    temp += a.arr[i*a.cols + k] * b.arr[k*b.cols + j]; // a.get(i, k) * b.get(k, j);
                arr[i * cols + j] = temp; //set(i, j, temp);
            }
        });
    }

    // this = a*b + c
    public void mulAndSum(Matrix a, Matrix b, Matrix c) {
        if (a.cols != b.rows || a.rows != rows || b.cols != cols)
            throw new MatrixException(a, b, this);
        if (rows != c.rows || cols != c.cols)
            throw new MatrixException(this, c);
        
        IntStream.range(0, rows).parallel().forEach((i) -> {
            for (int j = 0; j < cols; ++j) {
                int   idx  = i * cols + j;
                float temp = c.arr[idx];

                for (int k = 0; k < a.cols; ++k)
                    temp += a.arr[i * a.cols + k] * b.arr[k * b.cols + j];
                
                arr[idx] = temp;
            }
        });
    }

    public void mulAndSumAndApplyFunction(Matrix a, Matrix b, Matrix c, MathFunction f) {
        if (a.cols != b.rows || a.rows != rows || b.cols != cols)
            throw new MatrixException(a, b, this);
        if (rows != c.rows || cols != c.cols)
            throw new MatrixException(this, c);
        
        IntStream.range(0, rows).parallel().forEach((i) -> {
            for (int j = 0; j < cols; ++j) {
                int   idx  = i * cols + j;
                float temp = c.arr[idx];

                for (int k = 0; k < a.cols; ++k)
                    temp += a.arr[i * a.cols + k] * b.arr[k * b.cols + j];
                
                arr[idx] = f.compute(temp);
            }
        });
    }

    // this = this + other
    public void plus(Matrix other) {
        if (rows != other.rows || cols != other.cols)
            throw new MatrixException(this, other);

        for (int i = 0; i < rows*cols; ++i)
            arr[i] += other.arr[i];
    }

    // this = this - other
    public void minus(Matrix other) {
        if (rows != other.rows || cols != other.cols)
            throw new MatrixException(this, other);

        for (int i = 0; i < rows*cols; ++i)
            arr[i] -= other.arr[i];
    }

    // this = a - b
    public void minus(Matrix a, Matrix b) {
        if (a.cols != b.rows || a.rows != rows || b.cols != cols)
            throw new MatrixException(a, b, this);

        for (int i = 0; i < rows*cols; ++i)
            arr[i] = a.arr[i] - b.arr[i];
    }

    // this = (a - b).mulElemtnwilse( scalar)
    public void minusAndMul(Matrix a, Matrix b, float scalar) {
        if (a.cols != b.rows || a.rows != rows || b.cols != cols)
            throw new MatrixException(a, b, this);

        for (int i = 0; i < rows*cols; ++i)
            arr[i] = (a.arr[i] - b.arr[i]) * scalar;
    }

    public void apply(MathFunction f) {
        for (float it : arr)
            it = f.compute(it);
    }

    public void applyDerivative(MathFunction f) {
        for (float it : arr)
            it = f.derivative(it);
    }

    // dot product of this and other
    public float dot(Matrix other) {
        if (rows != other.rows || cols != other.cols)
            throw new MatrixException(this, other);

        float sum = 0.0f;
        for (int i = 0; i < rows*cols; ++i)
            sum += arr[i] * other.arr[i];
        return sum;
    }

    public void elementWiseMul(Matrix other) {
        for (int i = 0; i < rows*cols; ++i)
            arr[i] *= other.arr[i];
    }
}
