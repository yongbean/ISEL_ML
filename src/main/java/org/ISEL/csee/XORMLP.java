package org.ISEL.csee;

import java.util.*;

public class XORMLP {

    static class Perceptron {
        List<Double> weights;
        double bias;
        double output;
        double delta;

        public Perceptron(int inputSize) {
            weights = new ArrayList<>();
            for (int i = 0; i < inputSize; i++) {
                weights.add(Math.random() * 2 - 1); // 초기값 [-1, 1]
            }
            bias = Math.random() * 2 - 1;
        }

        public double forward(List<Double> input) {
            double sum = 0.0;
            for (int i = 0; i < input.size(); i++) {
                sum += input.get(i) * weights.get(i);
            }
            sum += bias;
            output = sigmoid(sum);
            return output;
        }

        private double sigmoid(double z) {
            return 1.0 / (1.0 + Math.exp(-z));
        }

        private double sigmoidDerivative() {
            return output * (1 - output);
        }

        public void updateWeights(List<Double> input, double learningRate) {
            for (int i = 0; i < weights.size(); i++) {
                weights.set(i, weights.get(i) - learningRate * delta * input.get(i));
            }
            bias -= learningRate * delta;
        }

        public void calculateOutputDelta(double target) {
            delta = (output - target) * sigmoidDerivative();
        }

        public void calculateHiddenDelta(double downstreamWeightSum) {
            delta = downstreamWeightSum * sigmoidDerivative();
        }
    }

    public static void main(String[] args) {
        List<List<Double>> inputs = Arrays.asList(
            Arrays.asList(0.0, 0.0),
            Arrays.asList(0.0, 1.0),
            Arrays.asList(1.0, 0.0),
            Arrays.asList(1.0, 1.0)
        );
        List<Double> targets = Arrays.asList(0.0, 1.0, 1.0, 0.0);

        List<Perceptron> hiddenLayer = Arrays.asList(
            new Perceptron(2),
            new Perceptron(2)
        );
        Perceptron outputNeuron = new Perceptron(2);

        double learningRate = 0.5;
        int epochs = 10000;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;

            for (int i = 0; i < inputs.size(); i++) {
                List<Double> x = inputs.get(i);
                double target = targets.get(i);

                // Forward
                List<Double> hiddenOutputs = new ArrayList<>();
                for (Perceptron h : hiddenLayer) {
                    hiddenOutputs.add(h.forward(x));
                }
                double yHat = outputNeuron.forward(hiddenOutputs);

                // Loss
                totalLoss += - (target * Math.log(yHat + 1e-8) + (1 - target) * Math.log(1 - yHat + 1e-8));

                // Backward
                outputNeuron.calculateOutputDelta(target);
                for (int j = 0; j < hiddenLayer.size(); j++) {
                    Perceptron h = hiddenLayer.get(j);
                    h.calculateHiddenDelta(outputNeuron.delta * outputNeuron.weights.get(j));
                }

                // Update
                outputNeuron.updateWeights(hiddenOutputs, learningRate);
                for (Perceptron h : hiddenLayer) {
                    h.updateWeights(x, learningRate);
                }
            }

            if (epoch % 1000 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d: Loss = %.4f%n", epoch, totalLoss);
            }
        }

        // Prediction
        System.out.println("\n▶ Final Predictions:");
        for (int i = 0; i < inputs.size(); i++) {
            List<Double> x = inputs.get(i);
            List<Double> hiddenOutputs = new ArrayList<>();
            for (Perceptron h : hiddenLayer) {
                hiddenOutputs.add(h.forward(x));
            }
            double yHat = outputNeuron.forward(hiddenOutputs);
            System.out.printf("Input: %s → Predicted = %.4f | Actual = %.0f%n", x, yHat, targets.get(i));
        }
    }
}
