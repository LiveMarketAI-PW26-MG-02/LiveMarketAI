public class Main {
    public static void main(String[] args) {
        double[] data = {1,2,3,4};
        double sum = 0;
        for(double d: data) sum += d;
        System.out.println(sum/data.length);
    }
}
