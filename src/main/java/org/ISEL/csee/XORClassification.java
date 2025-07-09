package org.ISEL.csee;

import java.util.*;

public class XORClassification {

//    public void run() {
//
//    }
    public static void main(String[] args) {
        // input x
        List<List<Double>> inputX = Arrays.asList(
                Arrays.asList(0.0, 0.0),
                Arrays.asList(0.0, 1.0),
                Arrays.asList(1.0, 0.0),
                Arrays.asList(1.0, 1.0)
        );

        // Y label
        List<Double> yLabel = Arrays.asList(0.0, 1.0, 1.0, 0.0);

        // single hidden layer (# of perceptrons in given hidden layer)
        List<Perceptron> hiddenLayer = Arrays.asList(
                new Perceptron(2),
                new Perceptron(2)
        );

        // output layer
        Perceptron outputPerceptron = new Perceptron(2);

        double learningRate = 0.5;
        int epochs = 10000;

        for(int epoch = 0; epoch < epochs; epoch++) {
            double error = 0.0;
            // 각각의 inputX에 대한 yhat 계산 => forward 값 계산
            for(int i = 0; i < inputX.size(); i++){
                List<Double> xLabel = inputX.get(i);
                double y = yLabel.get(i);

                // input layer -> hidden layer
                List<Double> hiddenOutputVal = new ArrayList<>();
                for(Perceptron hiddenPerceptron : hiddenLayer){
                    // hidden layer안의 perceptron의 개수만큼 data값 저장
                    hiddenOutputVal.add(hiddenPerceptron.forward(xLabel));
                }

                double yhat = outputPerceptron.forward(hiddenOutputVal);

                // 출력용 calc loss(error) => binary cross entropy 형태와 같음
                error += - (y * Math.log(yhat + 1e-8) + (1 - y) * Math.log(1 - yhat + 1e-8));

                // backpropagation 각 layer별로 연쇠작용으로 계산 => output layer + hidden layer 미분 값만 계산 필요
                // y label & yhat val에 대한 delta
                outputPerceptron.calcOutputDelta(y);
                for(int j = 0; j < hiddenOutputVal.size(); j++){
                    Perceptron h = hiddenLayer.get(j);
                    // current perceptron to next j perceptron on the  next layer
                    h.calcHiddenDelta(outputPerceptron.weights.get(j) * outputPerceptron.derivative);
                }

                // update
                // backpropagation에서 나온 미분 값 사용하여 weight & bias update
                outputPerceptron.updateWeights(hiddenOutputVal, learningRate);
                // hidden layer의 perceptron update
                for(Perceptron hiddenPerceptron : hiddenLayer){
                    hiddenPerceptron.updateWeights(xLabel, learningRate);
                }
            }

            if (epoch % 1000 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d: Loss = %.4f%n", epoch, error);
            }

        }

        // Prediction
        System.out.println("\n▶ Final Predictions:");
        for (int i = 0; i < inputX.size(); i++) {
            List<Double> x = inputX.get(i);
            List<Double> hiddenOutputs = new ArrayList<>();
            for (Perceptron h : hiddenLayer) {
                hiddenOutputs.add(h.forward(x));
            }
            double yHat = outputPerceptron.forward(hiddenOutputs);
            System.out.printf("Input: %s → Predicted = %.4f | Actual = %.0f%n", x, yHat, yLabel.get(i));
        }

    }
}


















