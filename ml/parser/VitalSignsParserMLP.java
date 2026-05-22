package ml.parser;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import ml.Matrix;

public class VitalSignsParserMLP implements Parser {  // Vibe coded
    private final int[] inputCols;
    private final int   labelCol;

    // Pass indices of the wanted columns 
    public VitalSignsParserMLP(int[] inputCols, int labelCol) {
        this.inputCols = inputCols;
        this.labelCol  = labelCol;
    }

    @Override
    public Matrix[] getDataset(String path) {
        List<Matrix> inputs = new ArrayList<>();
        try (Scanner scanner = new Scanner(new File(path))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.isEmpty()) 
                    continue;
                
                String[] parts = line.split(",");
                Matrix       x = new Matrix(inputCols.length, 1);
                
                for (int i = 0; i < inputCols.length; ++i) {
                    x.set(i, 0, Float.parseFloat(parts[inputCols[i]].trim()));
                }
                inputs.add(x);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        Matrix[] dataset = inputs.toArray(new Matrix[0]);
        minMaxScaling(dataset);

        return dataset;
    }

@Override
    public Matrix[] getAnnotations(String path) {
        List<Matrix> outputs = new ArrayList<>();
        try (Scanner scanner = new Scanner(new File(path))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.isEmpty()) continue;
                
                String[] parts = line.split(",");
                
                Matrix y = new Matrix(4, 1);
                y.zeroOut(); // Ensure all are 0.0 initially
                
                int rawClass = (int) Float.parseFloat(parts[labelCol].trim());
                
                // Arrays are 0 indexed, so  1 goes 0 and o on
                y.set(rawClass - 1, 0, 1.0f); 
                
                outputs.add(y);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return outputs.toArray(new Matrix[0]);
    }

    private void minMaxScaling(Matrix[] dataset) {
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
    }
}