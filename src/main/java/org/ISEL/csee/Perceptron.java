package org.ISEL.csee;

import java.util.ArrayList;
import java.util.List;

public class Perceptron {

    List<Double> weights;
    double bias;

    public Perceptron(int inputXSize) {
        // initialize weights & bias
        weights = new ArrayList<Double>();
        for (int i = 0; i < inputXSize; i++) {
            weights.add(Math.random() * 2 - 1); // [-1, 1]
        }
        bias = Math.random() * 2 - 1;
    }


}
