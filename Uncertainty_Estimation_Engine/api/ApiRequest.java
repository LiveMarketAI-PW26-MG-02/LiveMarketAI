package com.uncertainty.api;

import java.util.List;

/**
 * Incoming prediction request DTO.
 */
public class ApiRequest {
    private String modelName;
    private List<List<Double>> features;
    private boolean useCache = false;
    private String calibrationMethod = "none";

    public String getModelName() { return modelName; }
    public void setModelName(String modelName) { this.modelName = modelName; }
    public List<List<Double>> getFeatures() { return features; }
    public void setFeatures(List<List<Double>> features) { this.features = features; }
    public boolean isUseCache() { return useCache; }
    public void setUseCache(boolean useCache) { this.useCache = useCache; }
    public String getCalibrationMethod() { return calibrationMethod; }
    public void setCalibrationMethod(String calibrationMethod) { this.calibrationMethod = calibrationMethod; }
}
