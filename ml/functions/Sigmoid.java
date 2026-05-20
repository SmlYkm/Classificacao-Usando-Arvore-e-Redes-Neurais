package ml.functions;

public class Sigmoid implements MathFunction {
    public float compute(float n) {  // σ(x)        
        return 1.0f / (1.0f + (float)Math.exp((double)-n));
    }

    public float derivative(float sigCache) {  // σ'(x) = σ(x)⋅(1−σ(x)) Expects sigmoid cache already computed
        return sigCache*(1.0f-sigCache);
    }

}
