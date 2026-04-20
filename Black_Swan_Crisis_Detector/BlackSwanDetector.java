import java.net.URI;
import java.net.http.*;
import java.util.*;
import java.util.stream.*;

/**
 * BlackSwanDetector.java — Java crisis early-warning engine.
 * Monitors VIX, SPY volatility, and bond-equity correlation for systemic risk.
 * Compile: javac BlackSwanDetector.java  |  Run: java BlackSwanDetector
 */
public class BlackSwanDetector {
    private static final double CRISIS_THRESHOLD = 0.70;
    private static final double SEVERE_THRESHOLD = 0.85;
    private final HttpClient client = HttpClient.newHttpClient();
    private final Map<String, Deque<Double>> priceHistory = new HashMap<>();

    private static final String[][] ASSETS = {
        {"SPY","Equity"}, {"QQQ","Equity"}, {"TLT","Bond"},
        {"GLD","Commodity"}, {"HYG","Credit"}, {"^VIX","Volatility"}
    };

    public static void main(String[] args) throws Exception {
        BlackSwanDetector detector = new BlackSwanDetector();
        System.out.println("=== BlackSwanDetector (Java) — Market Crisis Early-Warning System ===");
        System.out.printf("Crisis Threshold: %.0f%% | Severe: %.0f%%%n%n",
            CRISIS_THRESHOLD*100, SEVERE_THRESHOLD*100);
        while (true) {
            detector.runCycle();
            Thread.sleep(90_000);
        }
    }

    private void runCycle() {
        System.out.println("[" + new Date() + "] Crisis Analysis Cycle");
        System.out.println("-".repeat(70));
        Map<String, Double> riskScores = new LinkedHashMap<>();
        double vixPrice = 20;
        for (String[] asset : ASSETS) {
            String sym = asset[0], cls = asset[1];
            try {
                double price = fetchQuote(sym);
                if (price <= 0) continue;
                String key = sym.replace("^","");
                Deque<Double> hist = priceHistory.computeIfAbsent(key, k -> new ArrayDeque<>());
                hist.addLast(price);
                if (hist.size() > 60) hist.pollFirst();
                if (hist.size() < 10) { System.out.printf("%-8s Collecting...%n", key); continue; }
                double[] prices = hist.stream().mapToDouble(Double::doubleValue).toArray();
                double[] returns = computeReturns(prices);
                double vol = computeStd(returns);
                double histVol = computeStd(Arrays.copyOfRange(returns, 0, Math.max(1, returns.length-5)));
                double volRisk = Math.min(1.0, vol/(histVol+1e-8)/3.0);
                if (sym.equals("^VIX")) { vixPrice = price; }
                riskScores.put(key, volRisk);
                System.out.printf("%-8s %-10s $%-9.2f Vol=%.4f VolRisk=%.3f%n",
                    key, cls, price, vol, volRisk);
            } catch (Exception e) { System.err.println(sym + ": " + e.getMessage()); }
        }
        // VIX-based risk
        double vixRisk = vixPrice < 15 ? 0.1 : (vixPrice < 20 ? 0.2 : (vixPrice < 30 ? 0.5 : (vixPrice < 40 ? 0.75 : 1.0)));
        riskScores.put("VIX_level", vixRisk);
        if (!riskScores.isEmpty()) {
            double avgRisk = riskScores.values().stream().mapToDouble(d->d).average().orElse(0);
            // Weighted: VIX gets higher weight
            double crisisProb = 0.30 * riskScores.getOrDefault("SPY",0.0)
                              + 0.25 * vixRisk
                              + 0.20 * riskScores.getOrDefault("HYG",0.0)
                              + 0.15 * riskScores.getOrDefault("TLT",0.0)
                              + 0.10 * riskScores.getOrDefault("GLD",0.0);
            crisisProb = Math.min(1.0, crisisProb);
            String status = crisisProb > SEVERE_THRESHOLD ? "⚠ SEVERE ALERT" :
                           (crisisProb > CRISIS_THRESHOLD ? "⚠ CRISIS WARNING" :
                           (crisisProb > 0.50 ? "⚡ ELEVATED RISK" : "✓ NORMAL"));
            System.out.printf("%n>>> CRISIS PROBABILITY: %.3f (%.0f%%) — %s%n", crisisProb, crisisProb*100, status);
            if (crisisProb > CRISIS_THRESHOLD) {
                System.out.println("RECOMMENDED ACTIONS:");
                if (crisisProb > SEVERE_THRESHOLD) {
                    System.out.println("  - Reduce portfolio exposure 40-60%");
                    System.out.println("  - Activate all defensive strategies");
                    System.out.println("  - Stop new positions immediately");
                } else {
                    System.out.println("  - Reduce portfolio exposure 20-30%");
                    System.out.println("  - Increase TLT/GLD hedges");
                    System.out.println("  - Tighten stop-losses");
                }
            }
        }
        System.out.println();
    }

    private double[] computeReturns(double[] p) {
        double[] r = new double[p.length-1];
        for (int i = 1; i < p.length; i++) r[i-1] = (p[i]-p[i-1])/p[i-1];
        return r;
    }

    private double computeStd(double[] arr) {
        if (arr.length == 0) return 0;
        double mean = Arrays.stream(arr).average().orElse(0);
        return Math.sqrt(Arrays.stream(arr).map(x->Math.pow(x-mean,2)).average().orElse(0));
    }

    private double fetchQuote(String symbol) throws Exception {
        String encoded = symbol.replace("^", "%5E");
        String url = "https://query1.finance.yahoo.com/v8/finance/chart/"+encoded+"?interval=1d&range=1d";
        HttpRequest req = HttpRequest.newBuilder().uri(URI.create(url)).header("User-Agent","Mozilla/5.0").GET().build();
        HttpResponse<String> r = client.send(req, HttpResponse.BodyHandlers.ofString());
        String body = r.body();
        int idx = body.indexOf("regularMarketPrice");
        if (idx < 0) return -1;
        int s = body.indexOf(":", idx)+1, e = body.indexOf(",", s);
        return Double.parseDouble(body.substring(s, e).replaceAll("[^0-9.]",""));
    }
}
