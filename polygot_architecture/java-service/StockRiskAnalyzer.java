import com.google.gson.Gson;
import com.google.gson.JsonObject;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

public class StockRiskAnalyzer {
    
    public static void main(String[] args) {
        try {
            // Read JSON input from stdin
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            StringBuilder jsonInput = new StringBuilder();
            String line;
            
            while ((line = reader.readLine()) != null) {
                jsonInput.append(line);
            }
            
            // Parse input
            Gson gson = new Gson();
            JsonObject input = gson.fromJson(jsonInput.toString(), JsonObject.class);
            
            String ticker = input.get("ticker").getAsString();
            double volatility = input.get("volatility").getAsDouble();
            double confidence = input.get("confidence").getAsDouble();
            String signal = input.get("signal").getAsString();
            
            System.err.println("Java: Analyzing risk for " + ticker + "...");
            
            // Calculate risk score (0-100)
            int riskScore = calculateRiskScore(volatility, confidence);
            
            // Determine risk level
            String riskLevel = getRiskLevel(riskScore);
            
            // Calculate position size recommendation (% of portfolio)
            double positionSize = calculatePositionSize(riskScore, signal);
            
            // Apply business rules
            String action = applyBusinessRules(signal, riskScore, volatility);
            
            // Create result
            Map<String, Object> result = new HashMap<>();
            result.put("service", "Java Risk Management");
            result.put("ticker", ticker);
            result.put("risk_score", riskScore);
            result.put("risk_level", riskLevel);
            result.put("recommended_action", action);
            result.put("position_size_pct", positionSize);
            
            Map<String, String> rules = new HashMap<>();
            rules.put("volatility_check", volatility > 30 ? "HIGH" : "NORMAL");
            rules.put("confidence_check", confidence > 0.7 ? "STRONG" : "WEAK");
            rules.put("diversification", "Apply 5% rule");
            result.put("business_rules", rules);
            
            // Output JSON
            String json = gson.toJson(result);
            System.out.println(json);
            
            System.err.println("Java: Risk analysis complete");
            
        } catch (Exception e) {
            System.err.println("Java Error: " + e.getMessage());
            e.printStackTrace(System.err);
            
            // Output error as JSON
            Map<String, String> error = new HashMap<>();
            error.put("service", "Java Risk Management");
            error.put("error", e.getMessage());
            System.out.println(new Gson().toJson(error));
        }
    }
    
    private static int calculateRiskScore(double volatility, double confidence) {
        // Higher volatility = higher risk
        // Lower confidence = higher risk
        int volRisk = (int) Math.min(volatility * 1.5, 60);
        int confRisk = (int) ((1 - confidence) * 40);
        return Math.min(volRisk + confRisk, 100);
    }
    
    private static String getRiskLevel(int score) {
        if (score < 30) return "LOW";
        if (score < 60) return "MEDIUM";
        return "HIGH";
    }
    
    private static double calculatePositionSize(int riskScore, String signal) {
        if (!signal.equals("BUY")) return 0.0;
        
        // Lower risk = larger position
        if (riskScore < 30) return 10.0;
        if (riskScore < 60) return 5.0;
        return 2.0;
    }
    
    private static String applyBusinessRules(String signal, int riskScore, double volatility) {
        // Override BUY if too risky
        if (signal.equals("BUY") && riskScore > 75) {
            return "HOLD (Risk too high)";
        }
        
        // Override SELL if volatility extreme
        if (signal.equals("SELL") && volatility > 40) {
            return "HOLD (Wait for stability)";
        }
        
        return signal;
    }
}
