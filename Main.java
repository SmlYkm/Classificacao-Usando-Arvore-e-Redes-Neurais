import ml.parser.VitalSignsParser;
import ml.neuralnet.MLP;
import ml.Matrix;

public class Main {
    public static void main(String[] args) {
        // 1. Initialize Parser (Inputs = cols 3,4,5. Target = col 7)
        int[] inputCols = {3, 4, 5};
        VitalSignsParser parser = new VitalSignsParser(inputCols, 7);
        
        Matrix[] dataset = parser.getDataset("02_treino_sinais_vitais_com_label.txt");
        Matrix[] annotations = parser.getAnnotations("02_treino_sinais_vitais_com_label.txt");

        // 2. Setup Neural Network
        int[] arch = {3, 8, 3, 1}; // 3 inputs, 8 hidden, 1 output
        MLP nn = new MLP(arch);
        nn.setDataset(dataset);
        nn.setAnnotations(annotations);
        nn.setLearningRate(0.01f);

        // 3. Train and Save
        System.out.println("Training...");
        nn.sgd(20000, 32); 

        System.out.println("Testing Neural Network...");
        int correct = 0;
        
        for (int i = 0; i < dataset.length; i++) {
            // 1. Feed the sample into the network
            nn.feedForward(dataset[i]);
            
            // 2. Get the raw prediction (e.g., 0.73)
            float rawPrediction = nn.getOutput().get(0, 0);
            
            // 3. DE-NORMALIZE: Multiply by 4 and round to nearest int
            int predictedClass = Math.round(rawPrediction * 4.0f);
            
            // Clamp it just in case the network outputs something extreme like 0.01
            predictedClass = Math.max(1, Math.min(4, predictedClass)); 
            
            // 4. De-normalize the actual answer to compare
            int actualClass = Math.round(annotations[i].get(0, 0) * 4.0f);
            
            if (predictedClass == actualClass) {
                correct++;
            }
        }
        
        float nnAccuracy = (float) correct / dataset.length * 100.0f;
        System.out.printf("Neural Network Accuracy: %.2f%%\n", nnAccuracy);


        nn.saveModel("vital_signs_weights.txt");

        // 4. Load later (even in a different execution)
        MLP loadedNN = new MLP(arch);
        loadedNN.loadModel("vital_signs_weights.txt");
    }
}