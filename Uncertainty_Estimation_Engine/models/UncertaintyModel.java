package com.uncertainty.models;

import java.util.*;

/**
 * Java representation of an uncertainty-aware model prediction.
 * Wraps predictions with decomposed uncertainty components.
 */
public class UncertaintyModel {

    private final String modelName;
    private final double[] predictions;
    private final double[] epistemicUncertainty;
    private final double[] aleatoricUncertainty;
    private final double[] totalUncertainty;
    private final Map<String, Object> metadata;

    public UncertaintyModel(
            String modelName,
            double[] predictions,
            double[] epistemicUncertainty,
            double[] aleatoricUncertainty) {
        this.modelName = modelName;
        this.predictions = Arrays.copyOf(predictions, predictions.length);
        this.epistemicUncertainty = Arrays.copyOf(epistemicUncertainty, epistemicUncertainty.length);
        this.aleatoricUncertainty = Arrays.copyOf(aleatoricUncertainty, aleatoricUncertainty.length);
        this.totalUncertainty = computeTotal(epistemicUncertainty, aleatoricUncertainty);
        this.metadata = new HashMap<>();
    }

    private double[] computeTotal(double[] epistemic, double[] aleatoric) {
        double[] total = new double[epistemic.length];
        for (int i = 0; i < epistemic.length; i++) {
            total[i] = Math.sqrt(epistemic[i] * epistemic[i] + aleatoric[i] * aleatoric[i]);
        }
        return total;
    }

    public double meanEpistemic() {
        return Arrays.stream(epistemicUncertainty).average().orElse(0.0);
    }

    public double meanAleatoric() {
        return Arrays.stream(aleatoricUncertainty).average().orElse(0.0);
    }

    public double meanTotalUncertainty() {
        return Arrays.stream(totalUncertainty).average().orElse(0.0);
    }

    public int size() { return predictions.length; }

    public String getModelName() { return modelName; }
    public double[] getPredictions() { return predictions; }
    public double[] getEpistemicUncertainty() { return epistemicUncertainty; }
    public double[] getAleatoricUncertainty() { return aleatoricUncertainty; }
    public double[] getTotalUncertainty() { return totalUncertainty; }
    public Map<String, Object> getMetadata() { return metadata; }

    public void addMetadata(String key, Object value) {
        metadata.put(key, value);
    }

    @Override
    public String toString() {
        return String.format(
            "UncertaintyModel{model='%s', n=%d, meanEpistemic=%.4f, meanAleatoric=%.4f}",
            modelName, predictions.length, meanEpistemic(), meanAleatoric()
        );
    }
}
