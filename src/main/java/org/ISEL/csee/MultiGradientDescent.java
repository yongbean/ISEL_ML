package org.ISEL.csee;

import java.util.ArrayList;
import java.util.List;

public class MultiGradientDescent {
    private final CostMethod costMethod;

    public MultiGradientDescent() {
        this.costMethod = new CostMethod();
    }

    public void run(List<List<Double>> wholeX, List<Double> y, float learningRate, List<Double> initial_w, int epochs) {
        List<Double> w = new ArrayList<>(initial_w);

        for (int epoch = 0; epoch < epochs; epoch++) {
            List<Double> diff = costMethod.calcMultiDiff(w, wholeX, y);
            List<Double> squared = costMethod.square(diff);
            double cost = squared.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

            List<Double> gradW = costMethod.graMultidW(diff, wholeX);
            double gradB = costMethod.gradB(diff);

            for (int i = 0; i < gradW.size(); i++) {
                w.set(i, w.get(i) - learningRate * gradW.get(i));
            }

            int biasIndex = w.size() - 1;
            w.set(biasIndex, w.get(biasIndex) - learningRate * gradB);

            if (epoch % 1000 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d: Cost=%.5f | w=%s | b=%.5f%n", epoch, cost, w.subList(0, w.size() - 1), w.get(biasIndex));
            }
        }

        System.out.println();
        int instanceIndex = (int) (Math.random() * wholeX.size());
        System.out.println("Instance index: " + instanceIndex);
        System.out.println("Input features: " + wholeX.get(instanceIndex));
        System.out.println("Resultant weights: " + w);

        double hx = 0;
        for (int i = 0; i < w.size(); i++) {
            hx += w.get(i) * wholeX.get(instanceIndex).get(i);
        }

        System.out.printf("Predicted H(x) = %.5f%n", hx);
        System.out.printf("Actual y = %.5f%n", y.get(instanceIndex));
    }

    public List<Double> trainOnly(List<List<Double>> X, List<Double> y, float lr, List<Double> w, int epochs) {
        double cost = 0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            List<Double> diff = costMethod.calcMultiDiff(w, X, y);
            List<Double> squared = costMethod.square(diff);
            cost = squared.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

            List<Double> gradW = costMethod.graMultidW(diff, X);
            double gradB = costMethod.gradB(diff);

            for (int i = 0; i < w.size(); i++) {
                double delta = (i == w.size() - 1) ? gradB : gradW.get(i);
                w.set(i, w.get(i) - lr * delta);
            }
        }
        System.out.println("Cost = " + cost);
        return w;
    }

}
