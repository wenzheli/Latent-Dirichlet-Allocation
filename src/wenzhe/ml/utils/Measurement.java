package wenzhe.ml.utils;

public class Measurement{
	
	/**
	 * calcualte the perplexity score for given documents. 
	 * 
	 * @param phi    topic-word distributions. 
	 * @param theta  doc-topic distributioins
	 * @param w      word for this corpus. 
	 * @param K      number of topics 
	 * @return       perplexity score. 
	 */
	public static double getPerplexityScore(double[][] phi, double[][] theta, int[][] w, int K){
			
		double log = 0;
		int count = 0;
		for (int m = 0; m < w.length; m++) {
			for (int n = 0; n < w[m].length; n++) {
				double sum = 0;
				for (int k = 0; k < K; k++) {
					sum += theta[m][k] * phi[k][w[m][n]];
				}
				log += Math.log(sum);
				count++;
			}
		}
		
		return Math.exp(-log / count)/5;
	}
}
