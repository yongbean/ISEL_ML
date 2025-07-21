package org.ISEL.csee;

import java.util.ArrayList;
import java.util.List;

public class Perceptron {

    List<Double> weights;
    double bias;
    double yhat;
    double derivative;

    public Perceptron(int inputXSize) {
        weights = new ArrayList<>();
        double limit = 1.0 / Math.sqrt(inputXSize);  // Xavier 범위 계산

        for (int i = 0; i < inputXSize; i++) {
            double w = (Math.random() * 2 * limit) - limit;  // [-limit, limit]
            weights.add(w);
        }

        bias = (Math.random() * 2 * limit) - limit;  // bias도 같은 방식으로 초기화
    }


    public double forward(List<Double> inputX) {
        double sum = 0.0;
        for (int i = 0; i < inputX.size(); i++) {
            sum += inputX.get(i) * weights.get(i);
        }
        sum += bias;
        yhat = sigmoid(sum);
        return yhat;
    }
    // used ReLU
//    public double forward(List<Double> input) {
//        this.input = input;
//        this.z = 0.0;
//        for (int i = 0; i < input.size(); i++) {
//            z += input.get(i) * weights.get(i);
//        }
//        z += bias;
//
//        // ReLU activation
//        outputVal = relu(z);
//        return outputVal;
//    }

    public double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public double sigmoidDerivative() {
        return yhat * (1 - yhat);
    }

    public double relu(double x) {
        return Math.max(0.0, x);
    }

    public double reluDerivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }

    // 이전 값의 Loss의 편차 * sigmoid에 의한 backpropagation 값
    public void calcOutputDelta(double yLabel) {
        derivative = (yhat - yLabel) * sigmoidDerivative();
    }

    public void calcHiddenDelta(double nodesWeightSum) {
        derivative = nodesWeightSum * sigmoidDerivative();
    }

    // Used ReLU
//    public void calcHiddenDelta(double downstreamDelta) {
//        derivative = downstreamDelta * reluDerivative(z); // z는 pre-activation 값
//    }

    public void updateWeights(List<Double> inputX, double learningRate) {
        for(int i = 0; i < weights.size(); i++) {
            weights.set(i, weights.get(i) - learningRate * derivative * inputX.get(i));
        }
        bias -= learningRate * derivative;
    }
}
