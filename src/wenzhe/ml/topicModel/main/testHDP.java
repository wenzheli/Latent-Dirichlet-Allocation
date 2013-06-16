package wenzhe.ml.topicModel.main;

/**
 *  @author Gregor Heinrich  (2006)
 * 	@author wenzhe  (2013)
 */
import java.util.Map;
import java.util.Random;

import wenzhe.ml.topicModel.datasets.Corpus;
import wenzhe.ml.topicModel.datasets.CorpusGenerator;

import wenzhe.ml.topicModel.learning.HDPModel;
import wenzhe.ml.topicModel.learning.LDAModel;
import wenzhe.ml.utils.Measurement;

public class testHDP {
	public static void main(String[] arhdp){
		
		int iterations = 10;
		
		String filePath = "data//review.json";
		CorpusGenerator gen = new CorpusGenerator(filePath, 1000);
		Corpus corpus = gen.createCorpus();
		
		// initialize parameters.  
		int K = 0;            // number of topics we initially have. 
		double alpha = 1.;
		double beta = 20;
		double gamma = 1.5;

		// HDP model, and run gibbs sampler. 
		HDPModel hdp = new HDPModel(K,  alpha, beta, gamma, new Random());
		hdp.init(corpus);   // initialize with corpus.  
		hdp.run(iterations);
		hdp.updateHyperParams();
		double[][] phi = hdp.getPhi();
		double[][] theta = hdp.getTheta();
		int k_curr = hdp.getTopicCount();
		int[][] w = hdp.getWords();
		double pspScore = Measurement.getPerplexityScore(phi, theta, w, k_curr);
		
		System.out.println("perplexity score is : " + pspScore);
		
	}
}
