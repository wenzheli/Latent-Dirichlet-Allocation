package wenzhe.ml.topicModel.main;

import java.util.Map;

import wenzhe.ml.topicModel.datasets.Corpus;
import wenzhe.ml.topicModel.datasets.CorpusGenerator;
import wenzhe.ml.topicModel.learning.LDAModel;
import wenzhe.ml.utils.Measurement;

public class testLDA {
	public static void main(String[] args){
		String testFile = "data//review.json";
		int k = 50;
		CorpusGenerator gen = new CorpusGenerator(testFile, 10000);
		Corpus corpus = gen.createCorpus();
		Map<String, Integer> vocab = corpus.getVocabulary();
		LDAModel lda = new LDAModel(0.1, 0.1, k);
		lda.init(corpus);
		System.out.println("running gibbs sampler....");
		lda.runCollapsedGibbs();
		
		// calculate the perplexity score
		double[][] theta = lda.getTheta();
		double[][] phi = lda.getBeta();
		int[][] w = lda.getWords();
		
		double pspScore = Measurement.getPerplexityScore(phi, theta, w, k);
		
		System.out.println("perplexity score is : " + pspScore);
	}

}
