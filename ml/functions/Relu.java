package ml.functions;

public class Relu implements MathFunction {
    public float compute(float n) {
        return (n > 0.0f) ? n : 0.0f; 
    }

    public float derivative(float n) {
        return (n > 0.0f) ? 1.0f : 0.0f; 
    }
}
