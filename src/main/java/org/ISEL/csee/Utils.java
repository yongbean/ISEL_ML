package org.ISEL.csee;

import java.util.List;
import java.util.stream.Collectors;

public class Utils {
    private final CostMethod costMethod;

    public Utils() {
        this.costMethod = new CostMethod();
    }

    public void startGradientDescent(List<Double> x, List<Double> y, float learningRate, float initial_w, float initial_b, int epochs) {
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
    }

    public void startMultiGradientDescent(List<List<Double>> wholeX, List<Double> y, float learningRate, float initial_w, float initial_b, int epochs) {
        float w = initial_w;
        float b = initial_b;
//        List<Double> diff = costMethod.calcDiff(w, b, x, y);
//        List<Double> squared = costMethod.square(diff);


    }
}
