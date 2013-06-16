package wenzhe.ml.topicModel.utils;



public class DirichletEst {
	/**
	 * fixpoint iteration on alpha using counts as input and estimating by Polya
	 * distribution directly. Eq. 55 in Minka (2003)
	 * 
	 * @param nmk
	 *            count data (documents in rows, topic associations in cols)
	 * @param nm
	 *            total counts across rows
	 * @param alpha
	 * @param alpha
	 */
	/**
	 * clamp value for small gamma and digamma values
	 */
	public static final double GAMMA = 0.577215664901532860606512090082;
	private static final double GAMMA_MINX = 1.e-12;
	private static final double DIGAMMA_MINNEGX = -1250;
	
	// limits for switching algorithm in digamma
	/** C limit */
	private static final double C_LIMIT = 49;
	/** S limit */
	private static final double S_LIMIT = 1e-5;
	public static double estimateAlphaMap(int[][] nmk, int[] nm, double alpha,
			double a, double b) {
		int i, m, k, iter = 200;
		double summk, summ;
		int M = nmk.length;
		int K = nmk[0].length;
		double alpha0 = 0;
		double prec = 1e-5;

		// alpha = ( a - 1 + alpha * [sum_m sum_k digamma(alpha + mnk) -
		// digamma(alpha)] ) /
		// ( b + K * [sum_m digamma(K * alpha + nm) - digamma(K * alpha)] )

		for (i = 0; i < iter; i++) {
			summk = 0;
			summ = 0;
			for (m = 0; m < M; m++) {
				summ += digamma(K * alpha + nm[m]);
				for (k = 0; k < K; k++) {
					summk += digamma(alpha + nmk[m][k]);
				}
			}
			summ -= M * digamma(K * alpha);
			summk -= M * K * digamma(alpha);
			alpha = (a - 1 + alpha * summk) / (b + K * summ);
			// System.out.println(alpha);
			// System.out.println(Math.abs(alpha - alpha0));
			if (Math.abs(alpha - alpha0) < prec) {
				return alpha;
			}
			alpha0 = alpha;
		}
		return alpha;
	}
	
	public static double digamma(double x) {
		// double y = digamma(x, 0);
		// System.out.println(y + " " + x);
		// return y;
		// }
		//
		// private static double digamma(double x, int level) {
		if (x >= 0 && x < GAMMA_MINX) {
			x = GAMMA_MINX;
		}
		if (x < DIGAMMA_MINNEGX) {
			// System.out.println("UNDERFLOW: level " + level + " x " + x);
			return digamma(DIGAMMA_MINNEGX + GAMMA_MINX);
		}
		if (x > 0 && x <= S_LIMIT) {
			// System.out.println("S_LIMIT: level " + level + " x " + x);
			// use method 5 from Bernardo AS103
			// accurate to O(x)
			return -GAMMA - 1 / x;
		}

		if (x >= C_LIMIT) {
			// System.out.println("C_LIMIT: level " + level + " x " + x);
			// use method 4 (accurate to O(1/x^8)
			double inv = 1 / (x * x);
			// 1 1 1 1
			// log(x) - --- - ------ + ------- - -------
			// 2 x 12 x^2 120 x^4 252 x^6
			return Math.log(x) - 0.5 / x - inv
					* ((1.0 / 12) + inv * (1.0 / 120 - inv / 252));
		}
		// if (level > 12000) {
		// System.out.println("recursion level " + level + " x " + x);
		// }
		// return digamma(x + 1, level + 1) - 1 / x;
		return digamma(x + 1) - 1 / x;
	}
}
