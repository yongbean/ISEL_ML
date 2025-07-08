package org.ISEL.csee;

import java.util.*;

public class MultinomialLogisticRegression {

    public List<Double> softmax(List<Double> z) {
        List<Double> result = new ArrayList<>();
        double max = Collections.max(z);  // for numerical stability
        double sum = 0.0;

        for (double val : z) {
            sum += Math.exp(val - max);
        }

        for (double val : z) {
            result.add(Math.exp(val - max) / sum);
        }

        return result;
    }

    public List<Double> oneHotEncode(int label, int numClasses) {
        List<Double> oneHot = new ArrayList<>(Collections.nCopies(numClasses, 0.0));
        if (label >= 0 && label < numClasses) {
            oneHot.set(label, 1.0);
        }
        return oneHot;
    }

    // weight 초기화 (labelType x feature)
    private List<List<Double>> initializeWeight(List<List<Double>> x, List<Double> y) {
        List<List<Double>> weight = new ArrayList<>();
        int labelType = 0;
        for (Double aDouble : y) {
            labelType = (int) Math.max(labelType, aDouble);
        }

        // labelType이 실제 클래스 수보다 1 작으므로 +1
        for (int i = 0; i <= labelType; i++) {
            List<Double> temp = new ArrayList<>();
            for (int j = 0; j < x.getFirst().size(); j++) {
                temp.add(Math.random() * 0.01);  // 작은 값으로 초기화
            }
            weight.add(temp);
        }
        return weight;
    }

    // z = W * x
    public List<Double> predict(List<List<Double>> weights, List<Double> x) {
        List<Double> hx = new ArrayList<>();
        for (List<Double> w : weights) {
            double sum = 0;
            for (int j = 0; j < w.size(); j++) {
                sum += w.get(j) * x.get(j);
            }
            hx.add(sum);
        }
        return hx;
    }

    public List<List<Double>> run(List<List<Double>> x, List<Double> y, double learningRate, int epochs) {
        List<List<Double>> weights = initializeWeight(x, y);
        int n = x.size();

        for(int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;

            for(int i = 0; i < x.size(); i++) {
                List<Double> xInstance = x.get(i);
                int yTrue = y.get(i).intValue();
                // Hx
                List<Double> yHat = predict(weights, xInstance);

                // softmax
                List<Double> soft = softmax(yHat);

                // 출력용 loss 계산 => -log(p_true)
                totalLoss += -Math.log(soft.get(yTrue) + 1e-15);

                // update weight
                for (int k = 0; k < weights.size(); k++) {
                    double error = soft.get(k) - (k == yTrue ? 1.0 : 0.0);  // \hat{y}_k - y_k

                    for (int j = 0; j < xInstance.size(); j++) {
                        double updatedWeight = weights.get(k).get(j) - learningRate * error * xInstance.get(j);
                        weights.get(k).set(j, updatedWeight);
                    }
                }
            }

            if (epoch % 10000 == 0 || epoch == epochs - 1) {
                double avgLoss = totalLoss / n;
                System.out.printf("Epoch: %d, Cost: %.6f%n", epoch, avgLoss);

            }
        }
        return weights;
    }

}
