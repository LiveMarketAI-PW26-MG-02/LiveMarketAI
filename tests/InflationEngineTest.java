package com.inflation;

import com.inflation.models.InflationForecast;
import com.inflation.models.InflationModelService;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;

/**
 * Unit tests for Java Inflation Indicators Engine.
 */
public class InflationEngineTest {

    @Test
    public void testInflationForecastCreation() {
        double[] fc  = {3.1, 3.2, 3.0, 2.9, 2.8};
        double[] lo  = {2.8, 2.9, 2.7, 2.6, 2.5};
        double[] hi  = {3.4, 3.5, 3.3, 3.2, 3.1};
        InflationForecast f = new InflationForecast("ARIMA", fc, lo, hi);
        assertEquals(5, f.horizon());
        assertEquals("ARIMA", f.getModelName());
        assertTrue(f.meanForecast() > 0);
    }

    @Test
    public void testForecastMeanIsWithinBounds() {
        double[] fc = {3.0, 3.1, 3.2};
        double[] lo = {2.5, 2.6, 2.7};
        double[] hi = {3.5, 3.6, 3.7};
        InflationForecast f = new InflationForecast("VAR", fc, lo, hi);
        assertTrue(f.meanForecast() >= f.minForecast());
        assertTrue(f.meanForecast() <= f.maxForecast());
    }

    @Test
    public void testModelServiceAvailableModels() {
        InflationModelService svc = new InflationModelService();
        List<String> models = svc.availableModels();
        assertFalse(models.isEmpty());
        assertTrue(models.contains("ARIMA"));
    }

    @Test
    public void testForecastCaching() {
        InflationModelService svc = new InflationModelService();
        InflationForecast f1 = svc.getForecast("ARIMA", 12);
        InflationForecast f2 = svc.getForecast("ARIMA", 12);
        assertEquals(f1.meanForecast(), f2.meanForecast(), 1e-9);
    }

    @Test
    public void testClearCache() {
        InflationModelService svc = new InflationModelService();
        svc.getForecast("LSTM", 6);
        svc.clearCache();
        // Should not throw after clearing
        InflationForecast f = svc.getForecast("LSTM", 6);
        assertNotNull(f);
    }
}
