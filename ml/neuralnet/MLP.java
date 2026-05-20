package ml.neuralnet;

import ml.Matrix;

public class MLP {
    private final int      nLayers;
    private final Matrix[] activations;
    private final Matrix[] weights; 
    private final Matrix[] biases;

    public MLP(int[] arch) {
        nLayers     = arch.length - 1;
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

    public MLP(MLP other) {
        nLayers     = other.nLayers;
        activations = new Matrix[nLayers+1];
        weights     = new Matrix[nLayers];
        biases      = new Matrix[nLayers];

        for (int i = 0; i < nLayers+1; ++i) {
            activations[i] = new Matrix(other.activations[i].getRows(), other.activations[i].getCols());
            weights[i]     = new Matrix(other.weights[i].getRows()    , other.weights[i].getCols());
            biases[i]      = new Matrix(other.biases[i].getRows()     , other.biases[i].getCols());
        }
    }

    public MLP createGradContainer() {
        return new MLP(this);
    }
}
