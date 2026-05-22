package ml.parser;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import ml.Matrix;

public class VitalSignsParserTree {  // Vibe coded
    private final int[] inputCols;
    private final int   labelCol;

    public VitalSignsParserTree(int[] inputCols, int labelCol) {
        this.inputCols = inputCols;
        this.labelCol  = labelCol;
    }

    public Matrix getCombinedDataset(String path) {
        List<float[]> rows = new ArrayList<>();

        try (Scanner scanner = new Scanner(new File(path))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.isEmpty()) 
                    continue;
                
                String[] parts = line.split(",");
                float[]  row   = new float[inputCols.length + 1];
                
                for (int i = 0; i < inputCols.length; ++i)
                    row[i] = Float.parseFloat(parts[inputCols[i]].trim());
                
                row[inputCols.length] = Float.parseFloat(parts[labelCol].trim());
                rows.add(row);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        if (rows.isEmpty()) 
            return new Matrix(0, 0);

        int numRows    = rows.size();
        int numCols    = rows.get(0).length;
        Matrix dataset = new Matrix(numRows, numCols);
        
        for (int i = 0; i < numRows; ++i) 
            for (int j = 0; j < numCols; ++j) 
                dataset.set(i, j, rows.get(i)[j]);
        return dataset;
    }
}