package ml.parser;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import ml.Matrix;

public class VitalSignsParser implements Parser {
    private final int[] inputCols;
    private final int   labelCol;

    // Pass indices of the wanted columns 
    public VitalSignsParser(int[] inputCols, int labelCol) {
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
        return inputs.toArray(new Matrix[0]);
    }

    @Override
    public Matrix[] getAnnotations(String path) {
        List<Matrix> outputs = new ArrayList<>();
        try (Scanner scanner = new Scanner(new File(path))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.isEmpty()) 
                    continue;
                
                String[] parts = line.split(",");
                Matrix       y = new Matrix(1, 1);
                
                float raw = Float.parseFloat(parts[labelCol].trim());
                y.set(0, 0, raw/4.0f);
                outputs.add(y);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return outputs.toArray(new Matrix[0]);
    }
}