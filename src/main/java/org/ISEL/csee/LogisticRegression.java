package org.ISEL.csee;

import java.util.*;

public class LogisticRegression {

    // sigmoid 함수
    public double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    // 예측값 계산: w^T x + b → sigmoid
    // H(x) 값을 sigmoid에 대입하여 결과 출력
    public double predict(List<Double> x, List<Double> w) {
        double z = 0;
        for (int i = 0; i < x.size(); i++) {
            // z += wx
            z += w.get(i) * x.get(i);
        }
        return sigmoid(z);
    }

    // 비용 함수: binary cross entropy (single sample)
    //loss(cost) = -(y * log(H(x)) + (1 - y) * log(1 - H(x)))
    public static double computeLoss(double y, double yHat) {
        return - (y * Math.log(yHat) + (1 - y) * Math.log(1 - yHat));
    }

    // gradient 계산 (single sample)
    public static List<Double> computeGrad(double y, double yHat, List<Double> x) {
        List<Double> gradW = new ArrayList<>();
        double error = yHat - y;
        for (double xi : x) {
            gradW.add(error * xi);
        }
        return gradW;
    }

    // 학습: gradient descent
    public List<Double> run(List<List<Double>> inputX, List<Double> y, List<Double> w, double learningRate, int epochs) {


        for(int epoch = 0; epoch < epochs; epoch++) {
            for(int j = 0; j < inputX.size(); j++) {
                List<Double> x = inputX.get(j);

                // find Hx (=yhat)
                double yHat = predict(x, w);

                // calc the change of w & b based on cost func(=get derivative version of w & b)
                List<Double> gradW = computeGrad(y.get(j), yHat, x);

                // 주어진 모든 weight & bias에 대해 1번씩 update
                // weight & bias 업데이트
                for (int i = 0; i < w.size(); i++) {
                    w.set(i, w.get(i) - learningRate * gradW.get(i));
                }

//                if (epoch % 10000 == 0 || epoch == epochs - 1) {
//                    System.out.printf("Prediction: %.4f | Loss: %.4f\n", yHat, computeLoss(y.get(j), yHat));
//                }
            }
        }
        return w;
    }
}
