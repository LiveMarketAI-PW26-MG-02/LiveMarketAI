package com.uncertainty.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

/**
 * Spring Boot application configuration for the Uncertainty Estimation Engine.
 */
@Configuration
@ConfigurationProperties(prefix = "uncertainty")
public class ApplicationConfig {

    private int mcSamples = 100;
    private double confidenceLevel = 0.1;
    private boolean cacheEnabled = true;
    private long cacheTtlSeconds = 3600;
    private String pythonApiUrl = "http://localhost:8000";
    private String defaultModel = "gp";

    public int getMcSamples() { return mcSamples; }
    public void setMcSamples(int v) { this.mcSamples = v; }
    public double getConfidenceLevel() { return confidenceLevel; }
    public void setConfidenceLevel(double v) { this.confidenceLevel = v; }
    public boolean isCacheEnabled() { return cacheEnabled; }
    public void setCacheEnabled(boolean v) { this.cacheEnabled = v; }
    public long getCacheTtlSeconds() { return cacheTtlSeconds; }
    public void setCacheTtlSeconds(long v) { this.cacheTtlSeconds = v; }
    public String getPythonApiUrl() { return pythonApiUrl; }
    public void setPythonApiUrl(String v) { this.pythonApiUrl = v; }
    public String getDefaultModel() { return defaultModel; }
    public void setDefaultModel(String v) { this.defaultModel = v; }
}
