package com.uncertainty.models;

import java.util.*;

/**
 * Factory for creating and managing uncertainty model instances.
 */
public class ModelFactory {

    public enum ModelType { BAYESIAN, ENSEMBLE, GP, DROPOUT }

    private static final Map<String, UncertaintyModel> registry = new HashMap<>();

    private ModelFactory() {}

    public static UncertaintyModel create(ModelType type, String name, int outputSize) {
        double[] preds = new double[outputSize];
        double[] epis = new double[outputSize];
        double[] alea = new double[outputSize];

        Random rng = new Random();
        for (int i = 0; i < outputSize; i++) {
            preds[i] = rng.nextGaussian();
            epis[i] = Math.abs(rng.nextGaussian()) * 0.1;
            alea[i] = Math.abs(rng.nextGaussian()) * 0.05;
        }

        UncertaintyModel model = new UncertaintyModel(name, preds, epis, alea);
        model.addMetadata("type", type.name());
        model.addMetadata("created_at", System.currentTimeMillis());
        registry.put(name, model);
        return model;
    }

    public static Optional<UncertaintyModel> get(String name) {
        return Optional.ofNullable(registry.get(name));
    }

    public static Set<String> listModels() {
        return Collections.unmodifiableSet(registry.keySet());
    }

    public static void remove(String name) {
        registry.remove(name);
    }
}
