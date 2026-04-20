package com.inflation.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

/**
 * Spring Boot application configuration for the Inflation Indicators Java service.
 */
@Configuration
@ConfigurationProperties(prefix = "inflation")
public class ApplicationConfig {

    private String pythonApiUrl   = "http://localhost:8000";
    private int    forecastHorizon = 12;
    private double fedTarget       = 2.0;
    private boolean cacheEnabled   = true;
    private long   cacheTtlSeconds = 3600;
    private String defaultModel    = "ARIMA";

    public String getPythonApiUrl()       { return pythonApiUrl; }
    public void setPythonApiUrl(String v) { this.pythonApiUrl = v; }
    public int getForecastHorizon()       { return forecastHorizon; }
    public void setForecastHorizon(int v) { this.forecastHorizon = v; }
    public double getFedTarget()          { return fedTarget; }
    public void setFedTarget(double v)    { this.fedTarget = v; }
    public boolean isCacheEnabled()       { return cacheEnabled; }
    public void setCacheEnabled(boolean v){ this.cacheEnabled = v; }
    public long getCacheTtlSeconds()      { return cacheTtlSeconds; }
    public void setCacheTtlSeconds(long v){ this.cacheTtlSeconds = v; }
    public String getDefaultModel()       { return defaultModel; }
    public void setDefaultModel(String v) { this.defaultModel = v; }
}
