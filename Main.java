import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import ml.Matrix;
import ml.tree.DecisionTree;

public class Main {
    public static void main(String[] args) throws Exception {
        // 1. Read the dataset
        List<float[]> dataList = new ArrayList<>();
        Scanner scanner = new Scanner(new File("02_treino_sinais_vitais_com_label.txt"));
        
        while (scanner.hasNextLine()) {
            String[] parts = scanner.nextLine().trim().split(",");
            float[] row = new float[parts.length];
            for (int i = 0; i < parts.length; i++) {
                row[i] = Float.parseFloat(parts[i]);
            }
            dataList.add(row);
        }
        scanner.close();

        // Populate Matrix
        int totalRows = dataList.size();
        int cols = dataList.get(0).length;
        Matrix dataset = new Matrix(totalRows, cols);
        for (int i = 0; i < totalRows; i++) {
            for (int j = 0; j < cols; j++) {
                dataset.set(i, j, dataList.get(i)[j]);
            }
        }

        // 2. We only want to use si3, si4, and si5 for splits (Columns 3, 4, 5)
        int[] allowedFeatures = {3, 4, 5}; 
        
        // 3. Train the Tree (Max depth 5 prevents it from memorizing the data)
        DecisionTree tree = new DecisionTree(5, allowedFeatures);
        tree.train(dataset);
        System.out.println("Tree built successfully!");

        // 4. Test it on the same dataset to see your base accuracy
        int correct = 0;
        for (int i = 0; i < totalRows; ++i) {
            // Extract a single row into a 1x8 matrix
            Matrix sample = new Matrix(1, cols);
            for (int j = 0; j < cols; ++j) sample.set(0, j, dataset.get(i, j));
            
            float prediction = tree.predict(sample);
            float actual = dataset.get(i, cols - 1); // Last col is the label
            
            if (Math.abs(prediction - actual) < 0.1f) {
                correct++;
            }
        }
        
        float accuracy = (float) correct / totalRows * 100.0f;
        System.out.printf("Accuracy on training data: %.2f%%\n", accuracy);
    }
}