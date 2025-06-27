package org.ISEL.csee;

import java.util.ArrayList;
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

    public void startMultiGradientDescent(List<List<Double>> wholeX, List<Double> y, float learningRate, List<Double> initial_w, int epochs) {
        List<Double> w = new ArrayList<>(initial_w);

        for(int epoch = 0; epoch < epochs; epoch++) {
            List<Double> diff = costMethod.calcMultDiff(w, wholeX, y);
            List<Double> squared = costMethod.square(diff);

            double cost = squared.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

            List<Double> gradW = costMethod.graMultdW(diff, wholeX);
            double gradB = costMethod.gradB(diff);

            // weight 업데이트
            for (int i = 0; i < gradW.size(); i++) {
                double newWeight = w.get(i) - learningRate * gradW.get(i);
                w.set(i, newWeight);
            }

            // bias 업데이트 (w의 마지막 원소)
            int biasIndex = w.size() - 1;
            w.set(biasIndex, w.get(biasIndex) - learningRate * gradB);

            if (epoch % 1000 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d: Cost=%.5f | w=%s | b=%.5f%n", epoch, cost, w.subList(0, w.size() - 1), w.get(biasIndex));
            }
        }

    }
}
