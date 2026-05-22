package ml.tree;


import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import ml.Matrix;

public class DecisionTree {
    private      Node   root;
    private final int   maxDepth;
    private final int[] allowedFeatures;

    private class Node {
        float   prediction;
        float   thresh;
        int     featureIdx;
        boolean isLeaf;
        Node    left;
        Node    right;
    
        public Node() {  // fallback
            left       = null;
            right      = null;
            isLeaf     = false;
            thresh     = 0.0f;
            featureIdx = 0;
            prediction = 0.0f;
        }
    }

    public DecisionTree(int maxDepth, int[] allowedFeatures) {
        this.maxDepth        = maxDepth;
        this.allowedFeatures = allowedFeatures;
    }

    public void train(Matrix dataset) {
        root = buildTree(dataset, 0);
    }

    private Node buildTree(Matrix data, int depth) {
        int  numRows = data.getRows();
        int  numCols = data.getCols();
        Node node    = new Node();

        Map<Float, Integer> classCounts = new HashMap<>();  // Compute purity and get most common class
        for (int i = 0; i < numRows; ++i) {
            float label = data.get(i, numCols-1);           // Class label => last col
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }

        float mostCommonClass = -1;
        int   maxCount        = -1;
        for (Map.Entry<Float, Integer> entry : classCounts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount        = entry.getValue();
                mostCommonClass = entry.getKey();
            }
        }

        if (depth >= maxDepth || classCounts.size() == 1 || numRows < 2) {  // base case
            node.isLeaf = true;
            node.prediction = mostCommonClass;
            return node;
        }  

        int   bestFeature   = -1;  // Find best split using gini
        float bestThreshold = 0;
        float bestGini      = Float.MAX_VALUE;

        for (int f : allowedFeatures) {  // Search only features allowed
            for (int i = 0; i < numRows; ++i) {  // Test every value as potential thresh
                float threshold   = data.get(i, f);
                float currentGini = calculateSplitGini(data, f, threshold);
            
                if (currentGini < bestGini) {
                    bestGini      = currentGini;
                    bestFeature   = f;
                    bestThreshold = threshold;
                }
            }
        }

        if (bestFeature == -1) {  // No split value found
            node.isLeaf     = true;
            node.prediction = mostCommonClass;
            return node;
        }

        Matrix[] split     = splitData(data, bestFeature, bestThreshold);
        Matrix   leftData  = split[0];
        Matrix   rightData = split[1];

        if (leftData.getRows() == 0 || rightData.getRows() == 0) {  // split is isolated to one side
            node.isLeaf     = true;
            node.prediction = mostCommonClass;
            return node;
        }

        node.featureIdx = bestFeature;
        node.thresh     = bestThreshold;
        node.left       = buildTree(leftData, depth + 1);  // Recursion
        node.right      = buildTree(rightData, depth + 1);  // Recursion

        return node;
    }


    private float calculateSplitGini(Matrix data, int featureIdx, float thresh) {
        int leftCount  = 0;
        int rightCount = 0;

        Map<Float, Integer> leftClasses  = new HashMap<>();
        Map<Float, Integer> rightClasses = new HashMap<>();

        int labelCol = data.getCols() - 1;
        int rows     = data.getRows();

        for (int i = 0; i < rows; ++i) {
            float val   = data.get(i, featureIdx);
            float label = data.get(i, labelCol);

            if (val <= thresh) {
                ++leftCount;
                leftClasses.put(label, leftClasses.getOrDefault(label, 0) + 1);
            } else {
                ++rightCount;
                rightClasses.put(label, rightClasses.getOrDefault(label, 0) + 1);
            }
        }

        float giniLeft  = calculateGini(leftClasses, leftCount);
        float giniRight = calculateGini(rightClasses, rightCount);
    
        return (leftCount * giniLeft + rightCount * giniRight) / data.getRows();  // Weighted avg Gini 
    }


    private float calculateGini(Map<Float, Integer> classCounts, int total) {
        if (total == 0)
            return 0;
        
        float impurity = 1.0f;
        for (int count : classCounts.values()) {
            float prob = (float) count / total;
            impurity  -= (prob*prob);
        }
        
        return impurity;
    }

    private Matrix[] splitData(Matrix data, int featureIndex, float threshold) {
        int numCols = data.getCols();
        List<float[]> leftList  = new ArrayList<>();
        List<float[]> rightList = new ArrayList<>();

        for (int i = 0; i < data.getRows(); ++i) {
            float[] row = new float[numCols];
            for (int j = 0; j < numCols; ++j)
                row[j] = data.get(i, j);

            if (row[featureIndex] <= threshold) {
                leftList.add(row);
            } else {
                rightList.add(row);
            }
        }

        Matrix leftMatrix = new Matrix(leftList.size(), numCols);
        for (int i = 0; i < leftList.size(); ++i) {
            float[] rowValues = leftList.get(i);

            for (int j = 0; j < numCols; ++j)
                leftMatrix.set(i, j, rowValues[j]);
        }
        
        Matrix rightMatrix = new Matrix(rightList.size(), numCols);
        for (int i = 0; i < rightList.size(); ++i) {
            float[] rowValues = rightList.get(i);
            
            for (int j = 0; j < numCols; ++j)
                rightMatrix.set(i, j, rowValues[j]);
        }

        return new Matrix[]{leftMatrix, rightMatrix};
    }

    // Predict a single row matrix
    public float predict(Matrix singleSample) {
        return traverseTree(singleSample, root);
    }

    private float traverseTree(Matrix sample, Node node) {
        if (node.isLeaf) 
            return node.prediction;
        
        boolean temp = (sample.get(0, node.featureIdx) <= node.thresh);
        return  temp ? traverseTree(sample, node.left) : traverseTree(sample, node.right);
    }
}