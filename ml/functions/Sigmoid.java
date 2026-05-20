package ml.functions;

public class Sigmoid implements MathFunction {
    public float compute(float n) {  // σ(x)
        return 1.0f / (1.0f + (float)Math.exp((double)-n));
    }

    public float derivative(float n) {  // σ'(x) = σ(x)⋅(1−σ(x))
        float sig = compute(n);
        return sig*(1.0f-sig);
    }

}
