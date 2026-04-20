public class Logic {
    public static void main(String[] args) {
        double[] data = {100,102,101,105,110};
        double sum = 0;
        for(double d : data) sum += d;
        System.out.println("Result: " + sum/data.length);
    }
}
