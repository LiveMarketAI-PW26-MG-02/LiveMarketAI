package com.uncertainty.metrics;

import java.util.HashMap;
import java.util.Map;

/**
 * Java implementation of uncertainty quality metrics.
 */
public class UncertaintyMetrics {

    public static double sharpness(double[] uncertainty) {
        double sum = 0;
        for (double u : uncertainty) sum += u;
        return sum / uncertainty.length;
    }

    public static double coverage(double[] y, double[] lower, double[] upper) {
        int covered = 0;
        for (int i = 0; i < y.length; i++) {
            if (y[i] >= lower[i] && y[i] <= upper[i]) covered++;
        }
        return (double) covered / y.length;
    }

    public static double meanIntervalWidth(double[] lower, double[] upper) {
        double total = 0;
        for (int i = 0; i < lower.length; i++) total += upper[i] - lower[i];
        return total / lower.length;
    }

    public static double gaussianNLL(double[] y, double[] mean, double[] std) {
        double nll = 0;
        for (int i = 0; i < y.length; i++) {
            double sigma2 = std[i] * std[i];
            nll += 0.5 * Math.log(2 * Math.PI * sigma2)
                 + Math.pow(y[i] - mean[i], 2) / (2 * sigma2);
        }
        return nll / y.length;
    }

    public static Map<String, Double> evaluateAll(
            double[] y, double[] mean, double[] std, double[] lower, double[] upper) {
        Map<String, Double> results = new HashMap<>();
        results.put("sharpness", sharpness(std));
        results.put("nll", gaussianNLL(y, mean, std));
        results.put("coverage", coverage(y, lower, upper));
        results.put("interval_width", meanIntervalWidth(lower, upper));
        return results;
    }
}
