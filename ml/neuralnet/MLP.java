package ml.neuralnet;

import ml.Matrix;
import ml.functions.MathFunction;
import ml.functions.Sigmoid;
import java.util.Random;

public class MLP extends FFNN {
    private final int          nLayers;
    private final Grad         gradient;
    private final Random       randomizer;
    private       float        rate;
    private       MathFunction activation;

    public MLP(int[] arch) {
        super(arch);
        activation  = new Sigmoid();  // fallback 
        nLayers     = arch.length - 1;
        gradient    = new Grad(this);
        randomizer  = new Random();
        rate        = 0.5f;           // fallback
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

    public float cost(int batchSize) {
        float total = 0.0f;

        for (int i = 0; i < batchSize; ++i) {
            feedForward(dataset[i]);
            Matrix output = activations[nLayers];
            output.minus(annotations[i]);
            total += output.dot(output);
        }

        return total / (float)batchSize; 
    }

    public void backprop() {
        int      batchLen = dataset.length;
        Matrix[] wGrad    = gradient.weights;
        Matrix[] bGrad    = gradient.biases;
        Matrix[] aGrad    = gradient.activations;

        for (int i = 0; i < nLayers; ++i) {
            wGrad[i].zeroOut();
            bGrad[i].zeroOut();
        }

        for (int i = 0; i < batchLen; ++i) {
            feedForward(dataset[i]);

            aGrad[nLayers].minus(              // dc    d  n (a-y)²   2
                getOutput(), annotations[i]    // -- = --  Σ -----  = -*(a-y) => 2/n can be ignored because of the learning rate
            );                                 // da   da i=1  n      n

            for (int j = 0; j < nLayers; ++j)  // Keep output, zero out the rest
                aGrad[j].zeroOut();
        
            for (int l = nLayers; l > 0; --l) {
                Matrix aCrnt = activations[l];
                Matrix aPrev = activations[l-1];
                Matrix wPrev = weights[l-1];
                int    rows  = wPrev.getRows();
                int    cols  = wPrev.getCols();
                                                                    //                  da
                for (int j = 0; j < rows; ++j) {                    //      a = σ(z) => -- = σ(z)⋅(1−σ(z)) = a⋅(1-a)
                    float a     = aCrnt.get(j);                     //                  dz
                    float dc_da = aGrad[l].get(j);                  //                      dc   dc da
                    float da_dz = activation.derivative(a);         //                  δ = -- = --*--
                    float delta = da_dz*dc_da;                      //                      db   da dz
                                                                    //           dz     d
                    bGrad[l-1].plusAt(j, 0, delta);              //           --- = ---(aPrev*w + b) = aPrev
                                                                    //           dw    dw
                    for (int k = 0; k < cols; ++k) {                // dc   da dc dz  |  dc   da dc dz  |  dz    d
                        float dz_dw = aPrev.get(k);                 // -- = --*--*--  |  -- = --*--*--  |  -- = --(a*w + b) = w
                        float w     = wPrev.get(j, k);              // dw   dz da dw  |  da   dz da da  |  da   da
                        wGrad[l-1].plusAt(j, k, da_dz*dc_da*dz_dw); 
                        aGrad[l-1].plusAt(k, 0, da_dz*dc_da*w);
                    }                                               
                }
            }
        }
    }


    public void learn() {
        Matrix wGrad[] = gradient.weights;
        Matrix bGrad[] = gradient.biases;
        Matrix aGrad[] = gradient.activations;

        for (int l = 0; l < nLayers; ++l) {
            weights[l].minusAndMul(wGrad[l], rate);
            biases[l].minusAndMul(bGrad[l], rate);
            activations[l].minusAndMul(aGrad[l], rate);
        }
    }

    private void shuffle() {
        if (dataset == null || annotations == null || dataset.length != annotations.length)
            return;
        for (int i = dataset.length; i > 0; --i) {
            int pos = randomizer.nextInt(i+1);

            Matrix temp  = dataset[i];   // sawp inputs
            dataset[i]   = dataset[pos];
            dataset[pos] = temp;

            temp             = annotations[i];  // Swap outputs
            annotations[i]   = annotations[pos];
            annotations[pos] = temp;
        }
    }

    private void runEpoch(int nBatches) {
        shuffle();
        for (int j = 0; j < nBatches; ++j) {
            backprop();
            learn();
        }
    }

    // Stochastic Gradient Descent
    public void sgd(int epochs, int batchlen) {
        int nSamples     = dataset.length;
        int nBatches     = nSamples / batchlen;
        int lastBatchlen = nSamples % batchlen;
    
        if (lastBatchlen > 0) {
            for (int i = 0; i < epochs; ++i) 
                runEpoch(nBatches);
            backprop();
            learn();
        } else {
            for (int i = 0; i < epochs; ++i) 
                runEpoch(nBatches);
        }
    }
}
