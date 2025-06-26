package org.ISEL.csee;

import java.util.ArrayList;
import java.util.List;

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

    // dJ/db
    public double gradB(List<Double> diff) {
        double sum = 0;
        for (double d : diff) {
            sum += d;
        }
        return (2.0 / diff.size()) * sum;
    }


}
