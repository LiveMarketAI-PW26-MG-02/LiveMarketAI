package com.uncertainty.api;

/**
 * Uncertainty prediction response DTO.
 */
public class ApiResponse {
    private String modelName;
    private double[] predictions;
    private double[] epistemicUncertainty;
    private double[] aleatoricUncertainty;
    private double[] totalUncertainty;
    private double[] lowerBound;
    private double[] upperBound;
    private double meanEpistemic;
    private double meanAleatoric;

    public String getModelName() { return modelName; }
    public void setModelName(String n) { this.modelName = n; }
    public double[] getPredictions() { return predictions; }
    public void setPredictions(double[] p) { this.predictions = p; }
    public double[] getEpistemicUncertainty() { return epistemicUncertainty; }
    public void setEpistemicUncertainty(double[] e) { this.epistemicUncertainty = e; }
    public double[] getAleatoricUncertainty() { return aleatoricUncertainty; }
    public void setAleatoricUncertainty(double[] a) { this.aleatoricUncertainty = a; }
    public double[] getTotalUncertainty() { return totalUncertainty; }
    public void setTotalUncertainty(double[] t) { this.totalUncertainty = t; }
    public double[] getLowerBound() { return lowerBound; }
    public void setLowerBound(double[] lb) { this.lowerBound = lb; }
    public double[] getUpperBound() { return upperBound; }
    public void setUpperBound(double[] ub) { this.upperBound = ub; }
    public double getMeanEpistemic() { return meanEpistemic; }
    public void setMeanEpistemic(double me) { this.meanEpistemic = me; }
    public double getMeanAleatoric() { return meanAleatoric; }
    public void setMeanAleatoric(double ma) { this.meanAleatoric = ma; }
}
