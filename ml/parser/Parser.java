package ml.parser;

import ml.Matrix;

public interface Parser {
    Matrix[] getDataset(String path);
    Matrix[] getAnnotations(String path);
}
