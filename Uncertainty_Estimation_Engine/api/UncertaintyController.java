package com.uncertainty.api;

import com.uncertainty.models.ModelFactory;
import com.uncertainty.models.UncertaintyModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;

/**
 * REST controller for uncertainty estimation endpoints.
 */
@RestController
@RequestMapping("/api/v1/uncertainty")
public class UncertaintyController {

    @Autowired
    private UncertaintyService uncertaintyService;

    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of("status", "ok", "version", "1.0.0"));
    }

    @GetMapping("/models")
    public ResponseEntity<Set<String>> listModels() {
        return ResponseEntity.ok(ModelFactory.listModels());
    }

    @PostMapping("/predict")
    public ResponseEntity<ApiResponse> predict(@RequestBody ApiRequest request) {
        try {
            ApiResponse response = uncertaintyService.predict(request);
            return ResponseEntity.ok(response);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().build();
        }
    }

    @PostMapping("/models/{name}")
    public ResponseEntity<Map<String, Object>> createModel(
            @PathVariable String name,
            @RequestParam(defaultValue = "BAYESIAN") String type,
            @RequestParam(defaultValue = "10") int outputSize) {
        ModelFactory.ModelType modelType = ModelFactory.ModelType.valueOf(type.toUpperCase());
        UncertaintyModel model = ModelFactory.create(modelType, name, outputSize);
        return ResponseEntity.ok(Map.of(
            "name", model.getModelName(),
            "mean_epistemic", model.meanEpistemic(),
            "mean_aleatoric", model.meanAleatoric()
        ));
    }
}
