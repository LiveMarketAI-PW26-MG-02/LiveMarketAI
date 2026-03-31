public class Attention {
    public static void main(String[] args){
        double[] data = {100,102,101,105,110};
        double sum = 0;
        double[] w = new double[data.length];

        for(int i=0;i<data.length;i++){
            w[i] = i+1;
            sum += w[i];
        }

        System.out.print("Java attention: ");
        for(int i=0;i<w.length;i++){
            System.out.print((w[i]/sum) + " ");
        }
    }
}
