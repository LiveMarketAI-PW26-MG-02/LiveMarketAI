import java.net.URI;
import java.net.http.*;
import java.util.*;

/**
 * AlphaEngine.java — Java alpha signal generator using multi-factor scoring.
 * Compile: javac AlphaEngine.java  |  Run: java AlphaEngine
 */
public class AlphaEngine {
    private static final String[] SYMBOLS = {"AAPL","MSFT","NVDA","GOOGL","AMZN","TSLA","SPY"};
    private static final double ALPHA_THRESHOLD = 0.25;
    // Weights
    private static final double W_RSI = 0.25, W_MOMENTUM = 0.35, W_VOLUME = 0.20, W_TREND = 0.20;
    private final HttpClient client = HttpClient.newHttpClient();
    private final Map<String, Deque<Double>> priceHistory = new HashMap<>();
    private final Map<String, Deque<Long>> volumeHistory = new HashMap<>();

    public static void main(String[] args) throws Exception {
        AlphaEngine engine = new AlphaEngine();
        System.out.println("=== AlphaEngine (Java) — Multi-Factor Alpha Signal Generator ===");
        System.out.printf("Alpha Threshold: %.2f%n%n", ALPHA_THRESHOLD);
        while (true) {
            engine.runCycle();
            Thread.sleep(120_000);
        }
    }

    private void runCycle() {
        System.out.println("[" + new Date() + "] Alpha Generation Cycle");
        System.out.printf("%-8s %-10s %-10s %-10s %-10s %-12s %-8s%n",
            "Symbol","Price","RSI","Momentum","Vol-Z","Alpha","Signal");
        System.out.println("-".repeat(72));
        for (String sym : SYMBOLS) {
            try {
                double[] data = fetchQuoteWithVolume(sym);
                if (data == null) continue;
                double price = data[0]; long volume = (long)data[1];
                Deque<Double> ph = priceHistory.computeIfAbsent(sym, k -> new ArrayDeque<>());
                Deque<Long> vh = volumeHistory.computeIfAbsent(sym, k -> new ArrayDeque<>());
                ph.addLast(price); vh.addLast(volume);
                if (ph.size() > 60) { ph.pollFirst(); vh.pollFirst(); }
                if (ph.size() < 20) { System.out.printf("%-8s Collecting...%n", sym); continue; }
                double[] prices = ph.stream().mapToDouble(Double::doubleValue).toArray();
                long[] vols = vh.stream().mapToLong(Long::longValue).toArray();
                double rsi = computeRSI(prices, 14);
                double momentum = prices[prices.length-1]/prices[prices.length-20] - 1;
                double volMean = Arrays.stream(vols).limit(20).average().orElse(1);
                double volZ = (vols[vols.length-1] - volMean) / (computeVolStd(vols) + 1);
                double trend = computeTrend(prices);
                double rsiScore = rsi > 70 ? -0.8 : (rsi < 30 ? 0.8 : (50-rsi)/50.0);
                double momScore = Math.max(-1, Math.min(1, momentum*15));
                double volScore = Math.max(-1, Math.min(1, volZ/3.0));
                double alpha = W_RSI*rsiScore + W_MOMENTUM*momScore + W_VOLUME*volScore + W_TREND*trend;
                alpha = Math.max(-1, Math.min(1, alpha));
                String signal = alpha > ALPHA_THRESHOLD ? "BUY" : (alpha < -ALPHA_THRESHOLD ? "SELL" : "HOLD");
                System.out.printf("%-8s $%-9.2f %-10.1f %-10.2f %-10.2f %-12.3f %-8s%n",
                    sym, price, rsi, momentum*100, volZ, alpha, signal);
            } catch (Exception e) { System.err.println(sym + ": " + e.getMessage()); }
        }
        System.out.println();
    }

    private double computeRSI(double[] prices, int period) {
        if (prices.length <= period) return 50;
        double gains = 0, losses = 0;
        for (int i = prices.length-period; i < prices.length; i++) {
            double d = prices[i]-prices[i-1];
            if (d > 0) gains += d; else losses -= d;
        }
        if (losses == 0) return 100;
        return 100 - 100/(1 + (gains/period)/(losses/period));
    }

    private double computeTrend(double[] prices) {
        int n = prices.length;
        double short_ = prices[n-1]/prices[Math.max(0,n-5)] - 1;
        double long_ = prices[n-1]/prices[Math.max(0,n-20)] - 1;
        return Math.max(-1, Math.min(1, (short_ > 0 && long_ > 0) ? 0.5 : (short_ < 0 && long_ < 0) ? -0.5 : 0));
    }

    private double computeVolStd(long[] vols) {
        double mean = Arrays.stream(vols).limit(20).average().orElse(0);
        return Math.sqrt(Arrays.stream(vols).limit(20).mapToDouble(v -> Math.pow(v-mean,2)).average().orElse(0));
    }

    private double[] fetchQuoteWithVolume(String symbol) throws Exception {
        String url = "https://query1.finance.yahoo.com/v8/finance/chart/"+symbol+"?interval=1d&range=1d";
        HttpRequest req = HttpRequest.newBuilder().uri(URI.create(url)).header("User-Agent","Mozilla/5.0").GET().build();
        HttpResponse<String> r = client.send(req, HttpResponse.BodyHandlers.ofString());
        String body = r.body();
        int pi = body.indexOf("regularMarketPrice");
        if (pi < 0) return null;
        int ps = body.indexOf(":", pi)+1, pe = body.indexOf(",", ps);
        double price = Double.parseDouble(body.substring(ps, pe).replaceAll("[^0-9.]",""));
        int vi = body.indexOf("regularMarketVolume");
        long vol = 0;
        if (vi >= 0) {
            int vs = body.indexOf(":", vi)+1, ve = body.indexOf(",", vs);
            try { vol = Long.parseLong(body.substring(vs, ve).replaceAll("[^0-9]","")); } catch (Exception e) {}
        }
        return new double[]{price, vol};
    }
}
