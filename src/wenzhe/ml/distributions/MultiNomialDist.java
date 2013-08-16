package wenzhe.ml.distributions;

/**
 * Multinomial distribution. We can sample the index
 * based on the input distributions. 
 * 
 * @author wenzhe
 *
 */
public class MultiNomialDist {
	private double[] p;
	
	public MultiNomialDist(double[] p){
		this.p = p;
	}
	
	/**
	 * sample the index based on the distributions.. 
	 * @return
	 */
	public int getSample(){
		int n = p.length;
		for (int i = 1; i < n; i++)
			p[i] += p[i-1];
		double u = Math.random() * p[n-1];
		int idx;
		for (idx = 0; idx < n; idx++){
			if (u < p[idx])
				break;
		}
		
		return idx;
	}
}
