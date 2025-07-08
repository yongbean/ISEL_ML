package org.ISEL.csee;

import java.util.*;

public class XORClassification {

//    public void run() {
//
//    }
    public static void main(String[] args) {
        // input x
        List<List<Double>> inputs = Arrays.asList(
                Arrays.asList(0.0, 0.0),
                Arrays.asList(0.0, 1.0),
                Arrays.asList(1.0, 0.0),
                Arrays.asList(1.0, 1.0)
        );

        // Y label
        List<Double> targets = Arrays.asList(0.0, 1.0, 1.0, 0.0);

        // single hidden layer (# of perceptrons in given hidden layer)
        List<Perceptron> hiddenLayer = Arrays.asList(
                new Perceptron(2),
                new Perceptron(2)
        );

        // output layer
        Perceptron outputNeuron = new Perceptron(2);

        double learningRate = 0.5;
        int epochs = 10000;


    }
}
