package com.inflation.api;

import com.inflation.models.InflationForecast;
import com.inflation.models.InflationModelService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.*;

/**
 * Java REST controller for inflation indicator endpoints.
 */
@RestController
@RequestMapping("/api/v1/inflation")
public class InflationController {

    @Autowired
    private InflationModelService modelService;

    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of("status", "ok", "service", "inflation-indicators"));
    }

    @GetMapping("/models")
    public ResponseEntity<List<String>> availableModels() {
        return ResponseEntity.ok(modelService.availableModels());
    }

    @GetMapping("/forecast/{model}")
    public ResponseEntity<Map<String, Object>> forecast(
            @PathVariable String model,
            @RequestParam(defaultValue = "12") int horizon) {
        InflationForecast fc = modelService.getForecast(model, horizon);
        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("model", fc.getModelName());
        resp.put("horizon_months", fc.horizon());
        resp.put("forecast", fc.getForecastValues());
        resp.put("lower_bound", fc.getLowerBound());
        resp.put("upper_bound", fc.getUpperBound());
        resp.put("mean_forecast", fc.meanForecast());
        return ResponseEntity.ok(resp);
    }

    @GetMapping("/snapshot")
    public ResponseEntity<Map<String, Object>> snapshot() {
        Random rng = new Random(System.currentTimeMillis() / 10000);
        Map<String, Object> snap = new LinkedHashMap<>();
        snap.put("cpi_yoy",       3.2 + rng.nextGaussian() * 0.3);
        snap.put("core_cpi_yoy",  2.8 + rng.nextGaussian() * 0.2);
        snap.put("ppi_yoy",       2.5 + rng.nextGaussian() * 0.5);
        snap.put("pce_yoy",       2.6 + rng.nextGaussian() * 0.2);
        snap.put("breakeven_10y", 2.3 + rng.nextGaussian() * 0.1);
        snap.put("regime",        "moderate");
        return ResponseEntity.ok(snap);
    }
}
