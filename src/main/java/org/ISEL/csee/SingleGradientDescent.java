package org.ISEL.csee;

import java.util.List;

public class SingleGradientDescent {
    private final CostMethod costMethod;

    public SingleGradientDescent() {
        this.costMethod = new CostMethod();
    }

    public void run(List<Double> x, List<Double> y, float learningRate, float initial_w, float initial_b, int epochs) {
        float w = initial_w;
        float b = initial_b;

        for (int epoch = 0; epoch < epochs; epoch++) {
            List<Double> diff = costMethod.calcDiff(w, b, x, y);
            List<Double> squared = costMethod.square(diff);
            double cost = squared.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

            double gradW = costMethod.gradW(diff, x);
            double gradB = costMethod.gradB(diff);

            w -= (float) (learningRate * gradW);
            b -= (float) (learningRate * gradB);

            if (epoch % 1000 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d: Cost=%.5f | w=%.5f | b=%.5f%n", epoch, cost, w, b);
            }
        }

        System.out.println();
        int instanceIndex = (int) (Math.random() * x.size());
        System.out.println("Instance index: " + instanceIndex);
        System.out.println("Input features: " + x.get(instanceIndex));
        System.out.println("Resultant weight: " + w);

        double hx = w * x.get(instanceIndex) + b;

        System.out.printf("Predicted H(x) = %.5f%n", hx);
        System.out.printf("Actual y = %.5f%n", y.get(instanceIndex));
    }

    public float[] trainOnly(List<Double> x, List<Double> y, float lr, float w, float b, int epochs) {
        double cost = 0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            List<Double> diff = costMethod.calcDiff(w, b, x, y);
            List<Double> squared = costMethod.square(diff);
            cost = squared.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

            double gradW = costMethod.gradW(diff, x);
            double gradB = costMethod.gradB(diff);

            w -= (float) (lr * gradW);
            b -= (float) (lr * gradB);
        }
        System.out.println("Cost = " + cost);
        return new float[]{w, b};
    }

}
