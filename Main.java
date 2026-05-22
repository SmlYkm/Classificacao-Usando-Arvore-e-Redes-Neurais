import ml.parser.VitalSignsParser;
import ml.neuralnet.MLP;
import ml.Matrix;

public class Main {
    public static void main(String[] args) {
        // 1. Initialize Parser
        int[] inputCols = {3, 4, 5};
        VitalSignsParser parser = new VitalSignsParser(inputCols, 7);
        
        Matrix[] dataset = parser.getDataset("02_treino_sinais_vitais_com_label.txt");
        Matrix[] annotations = parser.getAnnotations("02_treino_sinais_vitais_com_label.txt");

        // 2. Min-Max Scaling (Crucial for getting past 60%)
        int numFeatures = dataset[0].getRows();
        for (int f = 0; f < numFeatures; f++) {
            float min = Float.MAX_VALUE;
            float max = -Float.MAX_VALUE;
            for (int i = 0; i < dataset.length; i++) {
                float val = dataset[i].get(f, 0);
                if (val < min) min = val;
                if (val > max) max = val;
            }
            float range = max - min;
            if (range != 0) {
                for (int i = 0; i < dataset.length; i++) {
                    float val = dataset[i].get(f, 0);
                    dataset[i].set(f, 0, (val - min) / range);
                }
            }
        }

        // 3. The Automated Training Environment
        int[] arch = {3, 12, 1}; // Shallower, wider architecture
        float targetAccuracy = 90.0f;
        float bestAccuracy = 0.0f;
        int attempt = 1;
        int maxAttempts = 20; // Safety limit
        
        MLP bestNN = null;

        System.out.println("Starting Automated Search for >= " + targetAccuracy + "% Accuracy...");

        while (bestAccuracy < targetAccuracy && attempt <= maxAttempts) {
            System.out.println("\n--- Training Attempt " + attempt + " ---");
            
            // Create a fresh network with new random weights
            MLP nn = new MLP(arch);
            nn.setDataset(dataset);
            nn.setAnnotations(annotations);
            nn.setLearningRate(0.1f);

            // Train this instance (reduced epochs per attempt to save time)
            // If it's a "lucky" network, 5000 epochs is enough to show promise
            nn.sgd(1500, 32); 

            // Evaluate it using our helper method
            float currentAccuracy = evaluateAccuracy(nn, dataset, annotations);
            System.out.printf("Attempt %d reached: %.2f%%\n", attempt, currentAccuracy);

            // Keep track of the best one we've seen
            if (currentAccuracy > bestAccuracy) {
                bestAccuracy = currentAccuracy;
                bestNN = nn;
                System.out.printf(">>> New Best Model Found! (%.2f%%)\n", bestAccuracy);
            }

            attempt++;
        }

        // 4. Wrap up and Save
        System.out.println("\n=========================================");
        System.out.printf("Search Complete. Best Accuracy Achieved: %.2f%%\n", bestAccuracy);
        
        if (bestNN != null) {
            // Train the absolute best model a little bit more to squeeze out extra performance
            System.out.println("Fine-tuning the best model...");
            bestNN.setLearningRate(0.01f); // Lower learning rate for fine-tuning
            bestNN.sgd(5000, 32);
            
            float finalAcc = evaluateAccuracy(bestNN, dataset, annotations);
            System.out.printf("Final Fine-tuned Accuracy: %.2f%%\n", finalAcc);
            
            bestNN.saveModel("vital_signs_weights_best.txt");
        }
    }

    // --- Helper Method to keep the main loop clean ---
    private static float evaluateAccuracy(MLP nn, Matrix[] dataset, Matrix[] annotations) {
        int correct = 0;
        for (int i = 0; i < dataset.length; i++) {
            nn.feedForward(dataset[i]);
            float rawPrediction = nn.getOutput().get(0, 0);
            
            // De-normalize output
            int predictedClass = Math.round(rawPrediction * 4.0f);
            predictedClass = Math.max(1, Math.min(4, predictedClass)); 
            
            // De-normalize target
            int actualClass = Math.round(annotations[i].get(0, 0) * 4.0f);
            
            if (predictedClass == actualClass) {
                correct++;
            }
        }
        return (float) correct / dataset.length * 100.0f;
    }
}