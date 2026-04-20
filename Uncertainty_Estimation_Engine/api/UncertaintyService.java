package com.uncertainty.api;

import com.uncertainty.models.ModelFactory;
import com.uncertainty.models.UncertaintyModel;
import com.uncertainty.metrics.UncertaintyMetrics;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * Business logic for uncertainty estimation predictions.
 */
@Service
public class UncertaintyService {

    public ApiResponse predict(ApiRequest request) {
        Optional<UncertaintyModel> modelOpt = ModelFactory.get(request.getModelName());
        if (modelOpt.isEmpty()) {
            // Auto-create demo model if not registered
            ModelFactory.create(ModelFactory.ModelType.BAYESIAN, request.getModelName(),
                                request.getFeatures().size());
            modelOpt = ModelFactory.get(request.getModelName());
        }

        UncertaintyModel model = modelOpt.get();
        double[] preds = model.getPredictions();
        double[] ep    = model.getEpistemicUncertainty();
        double[] al    = model.getAleatoricUncertainty();
        double[] total = model.getTotalUncertainty();

        double[] lower = new double[preds.length];
        double[] upper = new double[preds.length];
        for (int i = 0; i < preds.length; i++) {
            lower[i] = preds[i] - 1.96 * total[i];
            upper[i] = preds[i] + 1.96 * total[i];
        }

        ApiResponse response = new ApiResponse();
        response.setModelName(model.getModelName());
        response.setPredictions(preds);
        response.setEpistemicUncertainty(ep);
        response.setAleatoricUncertainty(al);
        response.setTotalUncertainty(total);
        response.setLowerBound(lower);
        response.setUpperBound(upper);
        response.setMeanEpistemic(model.meanEpistemic());
        response.setMeanAleatoric(model.meanAleatoric());
        return response;
    }
}
