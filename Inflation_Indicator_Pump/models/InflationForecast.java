package com.inflation.models;

import java.util.Arrays;

/**
 * Represents a multi-step inflation forecast with confidence intervals.
 */
public class InflationForecast {

    private final String modelName;
    private final double[] forecastValues;
    private final double[] lowerBound;
    private final double[] upperBound;

    public InflationForecast(String modelName, double[] forecastValues,
                              double[] lowerBound, double[] upperBound) {
        this.modelName      = modelName;
        this.forecastValues = Arrays.copyOf(forecastValues, forecastValues.length);
        this.lowerBound     = Arrays.copyOf(lowerBound, lowerBound.length);
        this.upperBound     = Arrays.copyOf(upperBound, upperBound.length);
    }

    public double meanForecast() {
        return Arrays.stream(forecastValues).average().orElse(Double.NaN);
    }

    public double maxForecast() {
        return Arrays.stream(forecastValues).max().orElse(Double.NaN);
    }

    public double minForecast() {
        return Arrays.stream(forecastValues).min().orElse(Double.NaN);
    }

    public int horizon() { return forecastValues.length; }

    public String getModelName()      { return modelName; }
    public double[] getForecastValues() { return forecastValues; }
    public double[] getLowerBound()   { return lowerBound; }
    public double[] getUpperBound()   { return upperBound; }

    @Override
    public String toString() {
        return String.format("InflationForecast{model='%s', horizon=%d, mean=%.2f%%}",
                modelName, horizon(), meanForecast());
    }
}
