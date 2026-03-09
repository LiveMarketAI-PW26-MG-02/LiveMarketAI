import java.net.URI;
import java.net.http.*;
import java.util.*;

/**
 * RegimeDetector.java — Java component that fetches Yahoo Finance quotes
 * and applies volatility-based regime classification.
 * Compile: javac RegimeDetector.java
 * Run:     java RegimeDetector
 */
public class RegimeDetector {
    private static final String[] SYMBOLS = {"SPY", "QQQ", "GLD", "TLT"};
    private final HttpClient client = HttpClient.newHttpClient();
    private final Map<String, List<Double>> returnHistory = new HashMap<>();

    public static void main(String[] args) throws Exception {
        RegimeDetector detector = new RegimeDetector();
        System.out.println("=== RegimeDetector (Java) — Market Regime Analysis ===");
        while (true) {
            detector.runCycle();
            System.out.println("Next update in 60s... (Ctrl+C to stop)\n");
            Thread.sleep(60_000);
        }
    }

    private void runCycle() {
        System.out.println("\n[" + new Date() + "] Regime Analysis Cycle");
        System.out.println("-".repeat(60));
        List<String> regimes = new ArrayList<>();
        for (String sym : SYMBOLS) {
            try {
                double price = fetchQuote(sym);
                if (price <= 0) continue;
                List<Double> hist = returnHistory.computeIfAbsent(sym, k -> new ArrayList<>());
                if (!hist.isEmpty()) {
                    double ret = (price - hist.get(hist.size() - 1)) / hist.get(hist.size() - 1);
                    hist.add(price);
                    if (hist.size() > 60) hist.remove(0);
                    String regime = classifyRegime(hist);
                    double vol = computeVol(hist);
                    regimes.add(regime);
                    System.out.printf("%-6s | Price: %8.2f | Vol: %.4f | Regime: %s%n",
                            sym, price, vol, regime);
                } else {
                    hist.add(price);
                    System.out.printf("%-6s | Price: %8.2f | Collecting data...%n", sym, price);
                }
            } catch (Exception e) {
                System.err.println(sym + ": " + e.getMessage());
            }
        }
        if (!regimes.isEmpty()) {
            long crisis = regimes.stream().filter(r -> r.equals("CRISIS")).count();
            long high = regimes.stream().filter(r -> r.equals("HIGH_VOL")).count();
            String market = crisis >= regimes.size() * 0.4 ? "CRISIS" :
                            high >= regimes.size() * 0.5 ? "HIGH_VOL" : "NORMAL";
            System.out.println("\n>>> MARKET REGIME: " + market);
        }
    }

    private String classifyRegime(List<Double> prices) {
        double vol = computeVol(prices);
        if (vol < 0.005) return "LOW_VOL";
        if (vol < 0.012) return "MEDIUM_VOL";
        if (vol < 0.025) return "HIGH_VOL";
        return "CRISIS";
    }

    private double computeVol(List<Double> prices) {
        if (prices.size() < 2) return 0;
        List<Double> returns = new ArrayList<>();
        for (int i = 1; i < prices.size(); i++)
            returns.add((prices.get(i) - prices.get(i-1)) / prices.get(i-1));
        double mean = returns.stream().mapToDouble(d -> d).average().orElse(0);
        double variance = returns.stream().mapToDouble(d -> Math.pow(d - mean, 2)).average().orElse(0);
        return Math.sqrt(variance);
    }

    private double fetchQuote(String symbol) throws Exception {
        String url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol
                   + "?interval=1d&range=1d";
        HttpRequest req = HttpRequest.newBuilder().uri(URI.create(url))
            .header("User-Agent", "Mozilla/5.0").GET().build();
        HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
        String body = resp.body();
        int idx = body.indexOf("\"regularMarketPrice\":{\"raw\":");
        if (idx < 0) idx = body.indexOf("regularMarketPrice");
        if (idx < 0) return -1;
        int start = body.indexOf(":", idx) + 1;
        int end = body.indexOf(",", start);
        return Double.parseDouble(body.substring(start, end).replaceAll("[^0-9.]", ""));
    }
}
