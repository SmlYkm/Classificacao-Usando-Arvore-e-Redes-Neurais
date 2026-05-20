package ml.neuralnet;

import ml.Matrix;
import ml.functions.MathFunction;

public class MLP {
    private final int          nLayers;
    private final Matrix[]     activations;
    private final Matrix[]     weights; 
    private final Matrix[]     biases;
    private final MathFunction activation;

    public MLP(int[] arch, MathFunction activation) {
        nLayers     = arch.length - 1;
        activations = new Matrix[arch.length];
        weights     = new Matrix[nLayers];
        biases      = new Matrix[nLayers];
        this.activation = activation;

        activations[0] = new Matrix(arch[0], 1);
        for (int i = 1; i < arch.length; ++i) {
            activations[i] = new Matrix(arch[i], 1);
            weights[i-1]   = new Matrix(arch[i], arch[i-1]);
            biases[i-1]    = new Matrix(arch[i], 1);

            weights[i-1].randomize();
            biases[i-1].randomize();
        }
    }

    public MLP(MLP other, boolean copyValues) {
        nLayers     = other.nLayers;
        activation  = other.activation;
        activations = new Matrix[nLayers+1];
        weights     = new Matrix[nLayers];
        biases      = new Matrix[nLayers];

        activations[0] = new Matrix(other.activations[0].getRows(), other.activations[0].getCols());
        if (copyValues)
            activations[0].copy(other.activations[0]);
        for (int i = 1; i < nLayers+1; ++i) {
            activations[i] = new Matrix(other.activations[i].getRows(), other.activations[i].getCols());
            weights[i-1]   = new Matrix(other.weights[i-1].getRows()  , other.weights[i-1].getCols());
            biases[i-1]    = new Matrix(other.biases[i-1].getRows()   , other.biases[i-1].getCols());
            if (copyValues) {
                activations[i].copy(other.activations[i]);
                weights[i-1].copy(other.weights[i-1]);
                biases[i-1].copy(other.biases[i-1]);
            }
        }
    }

    public MLP createGradContainer() {
        return new MLP(this, false);
    }

    void feedForward(Matrix input) {
        activations[0].copy(input);

        for (int i = 0; i < nLayers; ++i) {
            Matrix w     = weights[i];  // References to w, a, b, for cleaner syntax
            Matrix b     = biases[i];
            Matrix a     = activations[i+1];
            Matrix aPrev = activations[i];

            a.mulAndSumAndApplyFunction(w, aPrev, b, activation);
        }
    }
}
