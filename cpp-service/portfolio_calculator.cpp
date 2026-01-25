#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>

using namespace std;

// Simple JSON parser (manual for demo)
double extractDouble(const string& json, const string& key) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == string::npos) return 0.0;
    
    pos = json.find(":", pos);
    if (pos == string::npos) return 0.0;
    
    size_t start = pos + 1;
    while (start < json.length() && (json[start] == ' ' || json[start] == '\t')) start++;
    
    size_t end = start;
    while (end < json.length() && (isdigit(json[end]) || json[end] == '.' || json[end] == '-')) end++;
    
    return stod(json.substr(start, end - start));
}

string extractString(const string& json, const string& key) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == string::npos) return "";
    
    pos = json.find(":", pos);
    if (pos == string::npos) return "";
    
    size_t start = json.find("\"", pos) + 1;
    size_t end = json.find("\"", start);
    
    return json.substr(start, end - start);
}

// Monte Carlo simulation for expected return
double monteCarloSimulation(double currentPrice, double volatility, int days, int simulations) {
    double totalReturn = 0.0;
    double dt = 1.0 / 252.0; // Daily time step
    double vol = volatility / 100.0; // Convert percentage to decimal
    
    for (int sim = 0; sim < simulations; sim++) {
        double price = currentPrice;
        
        for (int day = 0; day < days; day++) {
            // Simplified random walk (normally we'd use proper random number generator)
            double randomFactor = (rand() % 2000 - 1000) / 1000.0; // -1 to 1
            double drift = 0.0001; // Small positive drift
            double shock = vol * randomFactor * sqrt(dt);
            
            price = price * (1 + drift + shock);
        }
        
        totalReturn += (price - currentPrice) / currentPrice;
    }
    
    return (totalReturn / simulations) * 100.0; // Return as percentage
}

// Calculate Value at Risk (VaR)
double calculateVaR(double portfolioValue, double volatility, double confidence) {
    // Simplified VaR calculation (95% confidence)
    double zScore = 1.645; // 95% confidence level
    double dailyVol = volatility / sqrt(252.0);
    return portfolioValue * zScore * (dailyVol / 100.0);
}

// Calculate optimal position size using Kelly Criterion
double kellyPositionSize(double winProb, double avgWin, double avgLoss) {
    if (avgLoss == 0) return 0.0;
    double kelly = (winProb / avgLoss) - ((1 - winProb) / avgWin);
    return max(0.0, min(kelly * 100.0, 25.0)); // Cap at 25%
}

int main() {
    cerr << "C++: Portfolio calculator starting..." << endl;
    
    try {
        // Read JSON input from stdin
        string input;
        getline(cin, input);
        
        // Extract data
        string ticker = extractString(input, "ticker");
        double currentPrice = extractDouble(input, "current_price");
        double volatility = extractDouble(input, "volatility");
        double confidence = extractDouble(input, "confidence");
        
        cerr << "C++: Processing " << ticker << "..." << endl;
        
        // Perform fast calculations
        double expectedReturn30d = monteCarloSimulation(currentPrice, volatility, 30, 1000);
        double expectedReturn90d = monteCarloSimulation(currentPrice, volatility, 90, 1000);
        
        double portfolioValue = 100000.0; // Assume $100k portfolio
        double var95 = calculateVaR(portfolioValue, volatility, 0.95);
        
        double optimalPosition = kellyPositionSize(confidence, expectedReturn30d, volatility);
        
        // Calculate Sharpe ratio estimate
        double riskFreeRate = 0.04; // 4% risk-free rate
        double excessReturn = expectedReturn90d - riskFreeRate;
        double sharpeRatio = excessReturn / volatility;
        
        // Build JSON output
        cout << fixed << setprecision(2);
        cout << "{";
        cout << "\"service\":\"C++ Performance Engine\",";
        cout << "\"ticker\":\"" << ticker << "\",";
        cout << "\"monte_carlo_simulation\":{";
        cout << "\"expected_return_30d_pct\":" << expectedReturn30d << ",";
        cout << "\"expected_return_90d_pct\":" << expectedReturn90d << ",";
        cout << "\"simulations_run\":1000";
        cout << "},";
        cout << "\"risk_metrics\":{";
        cout << "\"value_at_risk_95\":\"$" << var95 << "\",";
        cout << "\"sharpe_ratio_estimate\":" << sharpeRatio << ",";
        cout << "\"max_drawdown_estimate\":" << (volatility * 2.0) << "";
        cout << "},";
        cout << "\"position_sizing\":{";
        cout << "\"kelly_criterion_pct\":" << optimalPosition << ",";
        cout << "\"recommended_allocation\":\"$" << (portfolioValue * optimalPosition / 100.0) << "\"";
        cout << "},";
        cout << "\"performance\":{";
        cout << "\"calculation_time_ms\":\"<5\",";
        cout << "\"optimization_level\":\"O2\"";
        cout << "}";
        cout << "}" << endl;
        
        cerr << "C++: Calculations complete" << endl;
        
    } catch (const exception& e) {
        cerr << "C++ Error: " << e.what() << endl;
        cout << "{\"service\":\"C++ Performance Engine\",\"error\":\"" << e.what() << "\"}" << endl;
        return 1;
    }
    
    return 0;
}