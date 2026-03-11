package com.uncertainty;

import com.uncertainty.models.ModelFactory;
import com.uncertainty.models.UncertaintyModel;
import com.uncertainty.metrics.UncertaintyMetrics;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Java Uncertainty Engine components.
 */
public class UncertaintyEngineTest {

    @Test
    public void testModelFactoryCreate() {
        UncertaintyModel model = ModelFactory.create(ModelFactory.ModelType.BAYESIAN, "testModel", 10);
        assertNotNull(model);
        assertEquals("testModel", model.getModelName());
        assertEquals(10, model.size());
    }

    @Test
    public void testMeanEpistemicNonNegative() {
        UncertaintyModel model = ModelFactory.create(ModelFactory.ModelType.ENSEMBLE, "ensModel", 20);
        assertTrue(model.meanEpistemic() >= 0.0);
    }

    @Test
    public void testTotalUncertaintyComputed() {
        double[] preds = {1.0, 2.0, 3.0};
        double[] epis  = {0.1, 0.2, 0.15};
        double[] alea  = {0.05, 0.1, 0.07};
        UncertaintyModel m = new UncertaintyModel("test", preds, epis, alea);
        double[] total = m.getTotalUncertainty();
        for (int i = 0; i < preds.length; i++) {
            double expected = Math.sqrt(epis[i]*epis[i] + alea[i]*alea[i]);
            assertEquals(expected, total[i], 1e-9);
        }
    }

    @Test
    public void testMetricsCoverage() {
        double[] y     = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] lower = {0.5, 1.5, 2.5, 3.5, 4.5};
        double[] upper = {1.5, 2.5, 3.5, 4.5, 5.5};
        double cov = UncertaintyMetrics.coverage(y, lower, upper);
        assertEquals(1.0, cov, 1e-9);
    }

    @Test
    public void testModelRegistryRemove() {
        ModelFactory.create(ModelFactory.ModelType.GP, "tempModel", 5);
        assertTrue(ModelFactory.listModels().contains("tempModel"));
        ModelFactory.remove("tempModel");
        assertFalse(ModelFactory.listModels().contains("tempModel"));
    }
}
