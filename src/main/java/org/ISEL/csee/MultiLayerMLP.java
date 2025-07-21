package org.ISEL.csee;

import java.util.*;

public class MultiLayerMLP {
    public static void main(String[] args) {
        List<List<Double>> inputX = Arrays.asList(
                Arrays.asList(0.0, 0.0),
                Arrays.asList(0.0, 1.0),
                Arrays.asList(1.0, 0.0),
                Arrays.asList(1.0, 1.0)
        );

        List<Double> yLabel = Arrays.asList(0.0, 1.0, 1.0, 0.0);

        int[] layerStructure = {2, 5, 5, 1}; // input:2 -> hidden1:3 -> hidden2:2 -> output:1
        List<List<Perceptron>> layers = new ArrayList<>();

        // Initialize layers
        for (int i = 1; i < layerStructure.length; i++) {
            List<Perceptron> layer = new ArrayList<>();
            for (int j = 0; j < layerStructure[i]; j++) {
                layer.add(new Perceptron(layerStructure[i - 1]));
            }
            layers.add(layer);
        }

        double learningRate = 0.1;
        int epochs = 100001;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;

            for (int i = 0; i < inputX.size(); i++) {
                List<Double> input = inputX.get(i);
                double target = yLabel.get(i);

                // ===== Forward Pass =====
                List<List<Double>> layerOutputs = new ArrayList<>();
                layerOutputs.add(input);

                for (List<Perceptron> layer : layers) {
                    List<Double> nextInput = new ArrayList<>();
                    for (Perceptron p : layer) {
                        nextInput.add(p.forward(input));
                    }
                    layerOutputs.add(nextInput);
                    input = nextInput;
                }

                double yHat = layerOutputs.get(layerOutputs.size() - 1).get(0);
                totalLoss += - (target * Math.log(yHat + 1e-8) + (1 - target) * Math.log(1 - yHat + 1e-8));

                // ===== Backward Pass =====
                // Output layer
                List<Perceptron> outputLayer = layers.get(layers.size() - 1);
                outputLayer.get(0).calcOutputDelta(target);

                // Hidden layers (역순)
                for (int l = layers.size() - 2; l >= 0; l--) {
                    List<Perceptron> currentLayer = layers.get(l);
                    List<Perceptron> nextLayer = layers.get(l + 1);

                    for (int j = 0; j < currentLayer.size(); j++) {
                        double sum = 0.0;
                        for (Perceptron next : nextLayer) {
                            sum += next.weights.get(j) * next.derivative;
                        }
                        currentLayer.get(j).calcHiddenDelta(sum);
                    }
                }

                // ===== Update Weights =====
                for (int l = 0; l < layers.size(); l++) {
                    List<Perceptron> layer = layers.get(l);
                    List<Double> prevOutput = layerOutputs.get(l);
                    for (Perceptron p : layer) {
                        p.updateWeights(prevOutput, learningRate);
                    }
                }
            }

            if (epoch % 10000 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d: Loss = %.6f\n", epoch, totalLoss);
            }
        }

        // Prediction
        System.out.println("\n▶ Final Predictions:");
        for (int i = 0; i < inputX.size(); i++) {
            List<Double> input = inputX.get(i);
            for (List<Perceptron> layer : layers) {
                List<Double> nextInput = new ArrayList<>();
                for (Perceptron p : layer) {
                    nextInput.add(p.forward(input));
                }
                input = nextInput;
            }
            System.out.printf("Input: %s → Predicted = %.4f | Actual = %.0f\n", inputX.get(i), input.get(0), yLabel.get(i));
        }
    }
}
