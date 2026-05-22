package ml.neuralnet;

import ml.Matrix;

abstract class FFNN {  // FFNN = feed forward neural net
    protected final Matrix[] activations;
    protected final Matrix[] weights; 
    protected final Matrix[] biases;

    // private class NeuralNetException extends RuntimeException{
    //     NeuralNetException(String message) {
    //         super(message);
    //     }
    // }


    FFNN(int[] arch) {
        int nLayers = arch.length - 1;
        activations = new Matrix[arch.length];
        weights     = new Matrix[nLayers];
        biases      = new Matrix[nLayers];

        activations[0] = new Matrix(arch[0], 1);
        for (int i = 1; i < arch.length; ++i) {
            activations[i] = new Matrix(arch[i], 1);
            weights[i-1]   = new Matrix(arch[i], arch[i-1]);
            biases[i-1]    = new Matrix(arch[i], 1);

            weights[i-1].randomize();
            biases[i-1].randomize();
        }
    }

    public abstract void feedForward(Matrix input);

    public abstract float cost(Matrix[] dataset, Matrix[] annotations, int batchSize);
}
