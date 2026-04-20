package com.inflation.models;

import org.springframework.stereotype.Service;
import java.util.*;

/**
 * Java service layer for inflation model management and result caching.
 */
@Service
public class InflationModelService {

    private final Map<String, InflationForecast> forecastCache = new LinkedHashMap<>();

    public InflationForecast getForecast(String modelName, int horizonMonths) {
        String key = modelName + "_" + horizonMonths;
        if (forecastCache.containsKey(key)) return forecastCache.get(key);

        InflationForecast fc = generateForecast(modelName, horizonMonths);
        forecastCache.put(key, fc);
        return fc;
    }

    private InflationForecast generateForecast(String modelName, int horizonMonths) {
        Random rng = new Random(modelName.hashCode());
        double[] values = new double[horizonMonths];
        double[] lower  = new double[horizonMonths];
        double[] upper  = new double[horizonMonths];
        double base = 3.0 + rng.nextGaussian() * 0.3;
        for (int i = 0; i < horizonMonths; i++) {
            values[i] = base + rng.nextGaussian() * 0.15 * Math.sqrt(i + 1);
            double margin = 0.3 * Math.sqrt(i + 1);
            lower[i]  = values[i] - margin;
            upper[i]  = values[i] + margin;
        }
        return new InflationForecast(modelName, values, lower, upper);
    }

    public List<String> availableModels() {
        return List.of("ARIMA", "VAR", "LSTM", "PHILLIPS_CURVE", "ENSEMBLE");
    }

    public void clearCache() { forecastCache.clear(); }
}
