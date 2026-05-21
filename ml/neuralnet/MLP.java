package ml.neuralnet;

import ml.Matrix;
import ml.functions.MathFunction;

public class MLP extends FFNN {
    private final MathFunction activation;
    private final int          nLayers;

    public MLP(int[] arch, MathFunction activation) {
        super(arch);
        this.activation = activation;
        this.nLayers    = arch.length - 1;
    }

    private class Grad {  // Auxiliary data class
        public final Matrix[] activations;
        public final Matrix[] weights; 
        public final Matrix[] biases;

        Grad(MLP net) {
            activations = new Matrix[net.activations.length];
            weights     = new Matrix[net.weights.length];
            biases      = new Matrix[net.biases.length];

            activations[0] = new Matrix(net.activations[0].getRows(), net.activations[0].getCols());
            for (int i = 1; i < activations.length; ++i) {
                activations[i] = new Matrix(net.activations[i].getRows(), net.activations[i].getCols());
                weights[i-1]   = new Matrix(net.weights[i-1].getRows()  , net.weights[i-1].getCols());
                biases[i-1]    = new Matrix(net.biases[i-1].getRows()   , net.biases[i-1].getCols());
            }
        }
    }

    public void feedForward(Matrix input) {
        activations[0].copy(input);

        for (int i = 0; i < nLayers; ++i) {
            Matrix w     = weights[i];  // References to w, a, b, for cleaner syntax
            Matrix b     = biases[i];
            Matrix a     = activations[i+1];
            Matrix aPrev = activations[i];

            a.mulAndSumAndApplyFunction(w, aPrev, b, activation);
        }
    }

    Matrix getOutput() {
        return activations[nLayers];
    }

    public float cost(Matrix[] dataset, Matrix[] annotations, int batchSize) {
        float total = 0.0f;

        for (int i = 0; i < batchSize; ++i) {
            feedForward(dataset[i]);
            Matrix output = activations[nLayers];
            output.minus(annotations[i]);
            total += output.dot(output);
        }

        return total / (float)batchSize; 
    }

    public void backprop(Matrix[] dataset, Matrix[] annotations) {
        int      batchLen = dataset.length;
        Grad     gradient = new Grad(this);
        Matrix[] wGrad    = gradient.weights;
        Matrix[] bGrad    = gradient.biases;
        Matrix[] aGrad    = gradient.activations;

        for (int i = 0; i < nLayers; ++i) {
            wGrad[i].zeroOut();
            bGrad[i].zeroOut();
        }

        for (int i = 0; i < batchLen; ++i) {
            feedForward(dataset[i]);
            // output = a*w + b
            //
            //                     1                               1
            // c = cost function = - * sum{(output-annotation)²} = - * sum{ [(a*w + b)-annotation]² }
            //                     n                               n
            //
            // dc    d 1                                    1
            // -- = -- - * sum{ [(a*w + b)-annotation]² } = - * 2 (a-annotation)
            // da   da n                                    n
            //
            // 2/n can be omited since it will be rescaled by the learning rate, thus
            //
            // dc
            // -- = a - annotation
            // da

            aGrad[nLayers].minusAndMul(// TODO: replace "2": 2 * (output - annotation) 
                activations[nLayers],  // output layer
                annotations[i], 
                2.0f
            );

            for (int j = 0; j < nLayers; ++j)  // Keep output layer
                aGrad[j].zeroOut();

            for (int j = nLayers; j > 0; --j) {
                Matrix currentActivations = activations[j];
                Matrix prevActivations    = activations[j-1];
                Matrix currentWeights     = weights[j-i];
                int rows = currentWeights.getRows();
                int cols = currentWeights.getCols();

                // a = currentActivations
                //
                //                     1
                // c = cost function = - * sum{(output-annotation)²}
                //                     n
                //
                // z = w*a + b
                //
                // dz   d
                // -- = -- (w*a + b) ===> dz = db
                // db   db
                //
                // da   da   
                // -- = -- = σ'(a)
                // db   dz
                //
                // dc           
                // -- = aGrad[j] 
                // da           
                //
                // dc   dc   dc   da
                // -- = -- = -- * -- = bGrad[j-1]
                // db   db   da   dz
                //
                // bGrad[j-1] = aGrad[j] * currentActivations.apply(Sigmoid.derivative)

                // dz
                // -- = prevActivations
                // dw
                //
                // dc   da   dc   dz
                // -- = -- * -- * --
                // dw   dz   da   dw
            }            
        }
    }
}
