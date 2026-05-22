import ml.parser.VitalSignsParserMLP;
import ml.parser.VitalSignsParserTree;
import ml.tree.DecisionTree;
import ml.neuralnet.MLP;
import ml.Matrix;

public class Main {
    public static void main(String[] args) {
        int    labelCol    = 7;          // y1
        int[]  inputCols   = {3, 4, 5};  // si3 si4 si5
        String datasetPath = "02_treino_sinais_vitais_com_label.txt"; 
        evaluateTree(inputCols, labelCol, datasetPath);
        evalueateNet(inputCols, labelCol);
    }

    private static float evaluateAccuracy(MLP nn, Matrix[] dataset, Matrix[] annotations) {
        int correct = 0;
        for (int i = 0; i < dataset.length; i++) {
            nn.feedForward(dataset[i]);
            Matrix output = nn.getOutput();
            
            int predictedClass = 1;
            float maxProb = -Float.MAX_VALUE;
            for (int j = 0; j < 4; j++) {
                if (output.get(j, 0) > maxProb) {
                    maxProb = output.get(j, 0);
                    predictedClass = j + 1; // +1 because classes are 1,2,3,4
                }
            }
            
            int actualClass = 1;  // Find idx
            for (int j = 0; j < 4; ++j) {
                if (annotations[i].get(j, 0) == 1.0f) {
                    actualClass = j + 1;
                    break;
                }
            }
            
            if (predictedClass == actualClass)
                ++correct;
        }
        return (float) correct / dataset.length * 100.0f;
    }

    private static void evalueateNet(int[] inputCols, int labelCol) {
        VitalSignsParserMLP parser = new VitalSignsParserMLP(inputCols, labelCol);
        
        Matrix[] dataset     = parser.getDataset("02_treino_sinais_vitais_com_label.txt");
        Matrix[] annotations = parser.getAnnotations("02_treino_sinais_vitais_com_label.txt");

        int[] arch = {3, 16, 4};
        float targetAccuracy = 90.0f;
        float bestAccuracy   = 0.0f;
        int   attempt        = 1;
        int   maxAttempts    = 20;
        
        MLP bestNN = null;

        while (bestAccuracy < targetAccuracy && attempt <= maxAttempts) {
            System.out.println("\n |/_ Training Attempt " + attempt + "|/_");
            
            MLP nn = new MLP(arch);
            nn.setDataset(dataset);
            nn.setAnnotations(annotations);
            nn.setLearningRate(0.1f);

            nn.sgd(1500, 32); 

            float currentAccuracy = evaluateAccuracy(nn, dataset, annotations);
            System.out.printf("Att %d acc: %.2f%%\n", attempt, currentAccuracy);

            if (currentAccuracy > bestAccuracy) {  // Keep best one
                bestAccuracy = currentAccuracy;
                bestNN = nn;
                System.out.printf("|/_|/_ Best one yet (%.2f%%)\n", bestAccuracy);
            }
            ++attempt;
        }

        System.out.println("\n|/_____________________________|");
        System.out.printf("I'ts over | acc: %.2f%%\n", bestAccuracy);
        
        if (bestNN != null) {
            System.out.println("Twaeking best a little");
            bestNN.setLearningRate(0.01f); 
            bestNN.sgd(5000, 32);
            
            float finalAcc = evaluateAccuracy(bestNN, dataset, annotations);
            System.out.printf("It's so over, nothing ever happens. final acc: %.2f%%\n", finalAcc);
            bestNN.saveModel("vital_signs_weights_best.txt");
        }
    }

    private static void evaluateTree(int inputCols[], int labelCol, String datasetPath) {
        VitalSignsParserTree treeParser = new VitalSignsParserTree(inputCols, labelCol);
        Matrix treeDataset = treeParser.getCombinedDataset(datasetPath);
        
        int[] allowedFeatures = {0, 1, 2};
        DecisionTree tree = new DecisionTree(5, allowedFeatures);
        
        System.out.println("|/_ =-= Tree accuracymaxxing =-= |/_");
        tree.train(treeDataset);
        
        int treeCorrect = 0;
        int numRows     = treeDataset.getRows();
        int numCols     = treeDataset.getCols();
        
        for (int i = 0; i < numRows; ++i) {
            Matrix sample = new Matrix(1, numCols);
            for (int j = 0; j < numCols; j++) 
                sample.set(0, j, treeDataset.get(i, j));
            
            float prediction = tree.predict(sample);
            float actual     = treeDataset.get(i, numCols - 1);
            if (Math.abs(prediction - actual) < 0.1f)
                ++treeCorrect;
        }
        float treeAccuracy = (float) treeCorrect / numRows * 100.0f;
        System.out.printf("Tree done accuracymaxxing - acc: %.2f%%)\n\n", treeAccuracy);
    }
}