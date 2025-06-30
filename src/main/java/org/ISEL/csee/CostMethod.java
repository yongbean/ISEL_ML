package org.ISEL.csee;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


/**
 *
 */

public class CostMethod {

    private double calcHx(float w, float b, double x) {
        return w * x + b;
    }
    // (y_pred - y_actual)
    public List<Double> calcDiff(float w, float b, List<Double> x, List<Double> y) {
        List<Double> diff = new ArrayList<>();
        for (int i = 0; i < x.size(); i++) {
            double pred = calcHx(w, b, x.get(i));
            diff.add(pred - y.get(i));
        }
        return diff;
    }

    private List<Double> calcMultiHx(List<List<Double>> x, List<Double> weights) {
        List<Double> result = new ArrayList<>();

        for (List<Double> row : x) {
            double dot = 0.0;
            for (int i = 0; i < row.size(); i++) {
                dot += row.get(i) * weights.get(i);
            }
            result.add(dot);
        }

        return result;
    }


    public List<Double> calcMultiDiff(List<Double> weights, List<List<Double>> x, List<Double> y) {
        List<Double> result = calcMultiHx(x, weights);
        List<Double> diff = new ArrayList<>();

        for (int i = 0; i < result.size(); i++) {
            diff.add(result.get(i) - y.get(i));
        }
        return diff;
    }

    public List<Double> square(List<Double> diff) {
        List<Double> result = new ArrayList<>();
        for (double d : diff) {
            result.add(d * d);
        }
        return result;
    }

    // dJ/dw
    public double gradW(List<Double> diff, List<Double> x) {
        double sum = 0;
        for (int i = 0; i < diff.size(); i++) {
            sum += diff.get(i) * x.get(i);
        }
        return (2.0 / x.size()) * sum;
    }

    public List<Double> graMultidW(List<Double> diff, List<List<Double>> x) {
        int m = x.size();                 // 데이터 개수
        int n = x.get(0).size();          // feature 개수
        List<Double> grads = new ArrayList<>(Collections.nCopies(n, 0.0));

        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int i = 0; i < m; i++) {
                sum += diff.get(i) * x.get(i).get(j);
            }
            grads.set(j, (2.0 / m) * sum);
        }

        return grads;
    }


    // dJ/db
    public double gradB(List<Double> diff) {
        double sum = 0;
        for (double d : diff) {
            sum += d;
        }
        return (2.0 / diff.size()) * sum;
    }
}
