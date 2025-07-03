package org.ISEL.csee;

import java.util.*;

public class LogisticRegression {

    // sigmoid 함수
    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    // 예측값 계산: w^T x + b → sigmoid
    // H(x) 값을 sigmoid에 대입하여 결과 출력
    public static double predict(List<Double> x, List<Double> w, double b) {
        double z = b;
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
    public static List<Double> computeGradW(double y, double yHat, List<Double> x) {
        List<Double> gradW = new ArrayList<>();
        double error = yHat - y;
        for (double xi : x) {
            gradW.add(error * xi);
        }
        return gradW;
    }

    public static double computeGradB(double y, double yHat) {
        return yHat - y;
    }

    // 학습: gradient descent (단일 샘플 기준)
    public static void trainStep(List<Double> x, double y, List<Double> w, double b, double learningRate) {
        // find Hx (=yhat)
        double yHat = predict(x, w, b);

        // calc the change of w & b based on cost func(=get derivative version of w & b)
        List<Double> gradW = computeGradW(y, yHat, x);
        double gradB = computeGradB(y, yHat);

        // 주어진 모든 weight & bias에 대해 1번씩 update
        // weight & bias 업데이트
        for (int i = 0; i < w.size(); i++) {
            w.set(i, w.get(i) - learningRate * gradW.get(i));
        }
        b -= learningRate * gradB;

        System.out.printf("Prediction: %.4f | Loss: %.4f\n", yHat, computeLoss(y, yHat));
    }

    public static void main(String[] args) {
        // 샘플 데이터: x = [2.0, 1.0], y = 1
        List<Double> x = Arrays.asList(2.0, 1.0);
        double y = 1.0;

        // 초기 가중치와 바이어스
        List<Double> w = new ArrayList<>(Arrays.asList(0.0, 0.0));
        double b = 0.0;

        double learningRate = 0.001;

        // 학습 10번만 반복
        for (int epoch = 0; epoch < 1000; epoch++) {
            System.out.printf("Epoch %d: ", epoch);
            trainStep(x, y, w, b, learningRate);
        }

        System.out.println("final weight w: " + w + " bias: " + b);
    }
}
