import ml.Matrix;
import ml.neuralnet.MLP;

public class Main {
    public static void main(String[] args) {
        // 1. Architecture: 2 inputs, 4 hidden neurons, 1 output
        int[] arch = {2, 4, 1};
        MLP nn = new MLP(arch);
        
        // nn.setLearningRate(0.5f); // Use this if you add the setter!

        // 2. Prepare XOR Dataset
        Matrix[] X = new Matrix[4];
        Matrix[] Y = new Matrix[4];

        for (int i = 0; i < 4; i++) {
            X[i] = new Matrix(2, 1);
            Y[i] = new Matrix(1, 1);
        }

        // 0 XOR 0 = 0
        X[0].set(0, 0, 0.0f); X[0].set(1, 0, 0.0f); Y[0].set(0, 0, 0.0f);
        // 0 XOR 1 = 1
        X[1].set(0, 0, 0.0f); X[1].set(1, 0, 1.0f); Y[1].set(0, 0, 1.0f);
        // 1 XOR 0 = 1
        X[2].set(0, 0, 1.0f); X[2].set(1, 0, 0.0f); Y[2].set(0, 0, 1.0f);
        // 1 XOR 1 = 0
        X[3].set(0, 0, 1.0f); X[3].set(1, 0, 1.0f); Y[3].set(0, 0, 0.0f);

        nn.setDataset(X);
        nn.setAnnotations(Y);

        // 3. Train
        System.out.println("Initial Cost: " + nn.cost(4));
        System.out.println("Training...");
        
        // 10,000 epochs, full batch (batch size 4)
        nn.sgd(10000, 4); 

        // 4. Test
        System.out.println("Final Cost: " + nn.cost(4));
        System.out.println("\nPredictions after training:");
        
        for (int i = 0; i < 4; i++) {
            nn.feedForward(X[i]);
            float prediction = nn.getOutput().get(0, 0);
            float target = Y[i].get(0, 0);
            
            System.out.printf("[%d] XOR [%d] = %.4f (Expected: %.1f)\n", 
                (int)X[i].get(0,0), (int)X[i].get(1,0), prediction, target);
        }
    }
}