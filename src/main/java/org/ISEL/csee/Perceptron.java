package org.ISEL.csee;

import java.util.ArrayList;
import java.util.List;

public class Perceptron {

    List<Double> weights;
    double bias;
    double yhat;
    double derivative;

    public Perceptron(int inputXSize) {
        // initialize weights & bias
        weights = new ArrayList<Double>();
        for (int i = 0; i < inputXSize; i++) {
            weights.add(Math.random() * 2 - 1); // [-1, 1]
        }
        bias = Math.random() * 2 - 1;
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

    public double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public double sigmoidDerivative() {
        return yhat * (1 - yhat);
    }

    // 이전 값의 Loss의 편차 * sigmoid에 의한 backpropagation 값
    public void calcOutputDelta(double yLabel) {
        derivative = (yhat - yLabel) * sigmoidDerivative();
    }

    public void calcHiddenDelta(double nodesWeightSum) {
        derivative = nodesWeightSum * sigmoidDerivative();
    }

    public void updateWeights(List<Double> inputX, double learningRate) {
        for(int i = 0; i < weights.size(); i++) {
            weights.set(i, weights.get(i) - learningRate * derivative * inputX.get(i));
        }
        bias -= learningRate * derivative;
    }
}
