package wenzhe.ml.topicModel.learning;

import static java.lang.Math.log;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;



import wenzhe.ml.topicModel.datasets.Corpus;

import wenzhe.ml.topicModel.datasets.Document;
import wenzhe.ml.topicModel.utils.DirichletEst;
import wenzhe.ml.utils.ArrayUtils;
import wenzhe.ml.utils.RandomSamplers;
import wenzhe.ml.utils.Vectors;

/**
 * HDP implementation, which supports infinit number of topics. 
 * For detailed information about HDP, please refer to paper [Tech 2005]
 * The implementation use as much code as possible from the LDA implementation.
 * For the implementation part, {@link arbylon.net/publications/ilda.pdf}‎
 * 
 * One important difference between LDA and HDP implementation is we use array for most
 * of part in LDA, but here we instead use list, since the number of topics
 * are incremented as we train our model. 
 * 
 * @author Gregor Heinrich  (2006)
 * @author wenzhe  (2013)
 *
 */
public class HDPModel {

	/**
	 * words for the corpus, it is two dimensional array, where the second dimension
	 * varies by document. 
	 */
	private int[][] w;
	
	/**
	 * K by V dimensional matrix, where K is the number if topics, and V is the number
	 * of vocabulary. It stores the summary of statistics. 
	 */
	private List<int[]> nkt;
	
	/**
	 * doc-topic distributions. It is M by K matrix, where each entry nmk_{i,j}
	 * is the total number of words assigned to topic j, from documen i.
	 */
	private List<Integer>[] nmk;
	
	/**
	 * sum of each row for variable {@link nkt}
	 */
	private List<Integer> nk;
	
	/**
	 * topic-word distributions. It is K by V matrix, where K is the 
	 * number of topics, and V is the number of words. 
	 */
	private double[][] phi;
	
	/**
	 * doc-topic distrubtions. It is M by K matrix, where M is the number
	 * of documents in the corpus, and K is the number of topics. 
	 */
	private double[][] theta;
	
	/**
	 * latent variables. It is M by V matrix, where each entry z_{i,j} 
	 * denotes the topic that assigned to the word j from document i. 
	 */
	private int[][] z;
	
	/**
	 * probability for each topic, this array used when we assign a topic
	 * to the latent variable z_{i,j} from multinomial distributions. 
	 */
	private double[] p;
	
	/**
	 * step to increase the sampling array
	 */
	public final int step = 10;
	
	/**
	 * precision of the 2nd-level DP
	 */
	private double alpha;
	
	/**
	 * mean of the 2nd-level DP = sample from 1st-level DP
	 */
	private ArrayList<Double> tau;
	
	/**
	 * parameter of root base measure (= component Dirichlet)
	 */
	private double beta;
	
	/**
	 * precision of root DP
	 */
	private double gamma;
	
	private SortedSet<Integer> indexingList;
	private List<Integer> activeList;
	
	/**
	 *  hyperparameters for DP and Dirichlet samplers
	 */
	double aalpha = 5;
	double balpha = 0.1;
	double abeta = 0.1;
	double bbeta = 0.1;
	double agamma = 5;
	double bgamma = 0.1;
	int R = 10;

	/**
	 * total number of tables
	 */
	private double T;
	
	/**
	 * random generator object.
	 */
	private Random rand;
	
	/**
	 * random sampler, which used to sample from different distributions. 
	 */
	private RandomSamplers samp;
	
	/** 
	 * number of iterations for gibbs sampling
	 */
	private int iter;
	
	/**
	 * current number of topics
	 */
	private int K;
	
	/**
	 * number of documents in this corpus
	 */
	private int M;

	/**
	 * vocabulary size
	 */
	private int V;
	
	/**
	 * boolean variables, which indicates this is HDP 
	 */
	private boolean isInited = false;
	private boolean isTopicFixed = false;
	private boolean isHyperParamFixed = false;
	
	/**
	 * get the words in this corpus. 
	 * @return   words 
	 */
	public int[][] getWords(){
		return this.w;
	}
	/**
	 * get the number of topics
	 * @return  topic size
	 */
	public int getTopicCount(){
		return this.K;
	}
	
	/**
	 * get the number of documents in the corpus
	 * @return    number of documents. 
	 */
	public int getDocumentCount(){
		return this.M;
	}
	
	/**
	 * get the vocabulary size
	 * @return   vocabulary size
	 */
	public int getVocabularySize(){
		return this.V;
	}
	
	/**
	 * get the phi 
	 * @return   phi
	 */
	public double[][] getPhi(){
		return this.phi;
	}
	
	/**
	 * get the theta, doc-topic distributions. 
	 * @return theta[][]
	 */
	public double[][] getTheta(){
		return this.theta;
	}
	
	/**
	 * get the latent variables Z, which is word assignment 
	 * for each word in the document. 
	 * @return  latent variables Z
	 */
	public int[][] getZ(){
		return this.z;
	}
	
	/**
	 * get the prior param alpha
	 * @return  alpha
	 */
	public double getAlpha(){
		return this.alpha;
	}
	
	/**
	 * get the prior param beta
	 * @return  beta
	 */
	public double getBeta(){
		return this.beta;
	}
	
	/**
	 * get the prior para gamma
	 * @return   gamma
	 */
	public double getGamma(){
		return this.gamma;
	}
	
		
	/**
	 * constructor. 
	 * @param K       number of topics 
	 * @param alpha   prior parameter
	 * @param beta    prior parameter
	 * @param gamma   prior for root of DP
	 * @param rand    
	 */
	public HDPModel(int K, double alpha,
			double beta, double gamma, Random rand) {
		
		this.K = K;
		this.alpha = alpha;
		this.beta = beta;
		this.gamma = gamma;
		if (gamma == 0) {
			this.isTopicFixed = true;
		}
		
		this.rand = rand;
		this.samp = new RandomSamplers(rand);
	}


	/**
	 * initialization, initialize all the temporarily variables and 
	 * latent variables. 
	 * 
	 * @param corpus    the corpus used for training the model. 
	 */
	public void init(Corpus corpus) {
		this.V = corpus.getTermCount();  // set the vocabulary size
		this.M = corpus.getDocCount();   // set the number of documents. 
		
		List<Document> documents = corpus.getDocuments();
    	w = new int[M][];
    	for (int i = 0 ; i < M; i++){
    		// construct each doc[i][] 
    		Document d = documents.get(i);
    		int count = d.getTotalTerms();
    		w[i] = new int[count];
    		for (int j = 0; j < count; j++)
    			w[i][j] = d.getTerm(j);
    	}
    	
    	// allocate the memory 
		nmk = new ArrayList[M];
		nkt = new ArrayList<int[]>();
		nk = new ArrayList<Integer>();
		z = new int[M][];
		for (int m = 0; m < M; m++) {
			nmk[m] = new ArrayList<Integer>();
			for (int k = 0; k < K; k++) {
				nmk[m].add(0);
			}
			z[m] = new int[w[m].length];
		}
		
		activeList = new ArrayList<Integer>();
		indexingList = new TreeSet<Integer>();
		tau = new ArrayList<Double>();
		for (int k = 0; k < K; k++) {
			activeList.add(k);
			nkt.add(new int[V]);
			nk.add(0);
			tau.add(1. / K);
		}
		tau.add(1. / K);
		p = new double[K + step];
		run(1);
		if (!isTopicFixed) {
			updateTau();
		}
		isInited = true;
	}

	

	/**
	 * run gibbs sampler. The sample is similar to the one for LDA
	 * The biggest difference here is, we sample the topic from K+1 topic,
	 * if we sample the topic K+1, we explicitly create the new topic
	 * and update corresponding parameters. 
	 * 
	 * @param iterations   number of iterations. 
	 */
	public void run(int iterations) {

		for (iter = 0; iter < iterations; iter++) {
			System.out.println(iter);
			for (int m = 0; m < M; m++) {
				for (int n = 0; n < w[m].length; n++) {
					// sampling z
					int k_curr, k_prev = -1;
					int t = w[m][n];
					if (isInited) {
						k_curr = z[m][n];
						// remove the current word, and update the parameters. 
						// same as LDA model. 
						nmk[m].set(k_curr, nmk[m].get(k_curr) - 1);
						nkt.get(k_curr)[t]--;
						nk.set(k_curr, nk.get(k_curr) - 1);
						k_prev = k_curr;
					}
					// compute the probablities for each topic j
					// p(z=j|*). then select the new topic as multinomial distributions.
					// same as LDA model, but we select from K+1 topic.
					double sum = 0;
					for (int kk = 0; kk < K; kk++) {
						k_curr = activeList.get(kk);
						p[kk] = (nmk[m].get(k_curr) + alpha * tau.get(k_curr)) * //
								(nkt.get(k_curr)[t] + beta) / (nk.get(k_curr) + V * beta);
						sum += p[kk];
					}
					// sample new index based on the p;
					if (!isTopicFixed) {
						p[K] = alpha * tau.get(K) / V;
						sum += p[K];
					}
					double u = rand.nextDouble();
					u *= sum;
					sum = 0;
					int kk = 0;
					for (; kk < K + 1; kk++) {
						sum += p[kk];
						if (u <= sum) {
							break;
						}
					}
					
					// if the topic number is already in the topic set.. update!
					// same as LDA model update.
					if (kk < K) {
						k_curr = activeList.get(kk);
						// update the current topic, and other parameters correspondingly. 
						z[m][n] = k_curr;
						nmk[m].set(k_curr, nmk[m].get(k_curr) + 1);
						nkt.get(k_curr)[t]++;
						nk.set(k_curr, nk.get(k_curr) + 1);
					} else {
						z[m][n] = createTopic(m, t);
						updateTau();
						System.out.println("create a new topic. K = " + K);
					}
					if (isInited && nk.get(k_prev) == 0) {
						// remove the object not the index
						activeList.remove((Integer) k_prev);
						indexingList.add(k_prev);
						K--;
						System.out.println("K = " + K);
						updateTau();
					}
				} 
			} 
			
			if (!isTopicFixed) {
				updateTau();
			}
			if (iter > 10 && !isHyperParamFixed) {
				updateHyperParams();
			}
		} 
	}

	

	/**
	 * add a topic to the current active topic list. 
	 * OR remove it from list. 
	 * 
	 * @param m     document m
	 * @param t     term t
	 * @return      new index;
	 */
	private int createTopic(int m, int t) {
		int k;
		if (indexingList.size() > 0) {
			k = indexingList.first();
			indexingList.remove(k);
			activeList.add(k);
			nmk[m].set(k, 1);
			nkt.get(k)[t] = 1;
			nk.set(k, 1);
		} else {
			k = K;
			for (int i = 0; i < M; i++) {
				nmk[i].add(0);
			}
			activeList.add(K);
			nmk[m].set(K, 1);
			nkt.add(new int[V]);
			nkt.get(K)[t] = 1;
			nk.add(1);
			tau.add(0.);
		}
		K++;
		if (p.length <= K) {
			p = new double[K + step];
		}
		return k;
	}


	/**
	 * update tau params, which is root of the DP 
	 */
	private void updateTau() {
		double[] mk = new double[K + 1];
		for (int kk = 0; kk < K; kk++) {
			int k = activeList.get(kk);
			for (int m = 0; m < M; m++) {
				if (nmk[m].get(k) > 1) {
					mk[kk] += samp.randAntoniak(alpha * tau.get(k), //
							nmk[m].get(k));
				} else {
					mk[kk] += nmk[m].get(k);
				}
			}
		}
		// number of tables
		T = Vectors.sum(mk);
		mk[K] = gamma;
		double[] tt = samp.randDir(mk);
		for (int kk = 0; kk < K; kk++) {
			int k = activeList.get(kk);
			tau.set(k, tt[kk]);
		}
		tau.set(K, tt[K]);
	}

	/**
	 * upate hyperparameters for HDP. 
	 */
	public void updateHyperParams() {
		for (int r = 0; r < R; r++) {
			double eta = samp.randBeta(gamma + 1, T);
			double bloge = bgamma - log(eta);
			double pie = 1. / (1. + (T * bloge / (agamma + K - 1)));
			int u = samp.randBernoulli(pie);
			gamma = samp.randGamma(agamma + K - 1 + u, 1. / bloge);

			double qs = 0;
			double qw = 0;
			for (int m = 0; m < M; m++) {
				qs += samp.randBernoulli(w[m].length / (w[m].length + alpha));
				qw += log(samp.randBeta(alpha + 1, w[m].length));
			}
		
			alpha = samp.randGamma(aalpha + T - qs, 1. / (balpha - qw));
		}
		int[] ak = (int[]) ArrayUtils.asPrimitiveArray(nk);
		int[][] akt = new int[K][V];
		for (int k = 0; k < K; k++) {
			akt[k] = nkt.get(k);
		}
		beta = DirichletEst
				.estimateAlphaMap(akt, ak, beta, abeta, bbeta);
		
		phi = new double[K][V];
		theta = new double[M][K];
		for (int k = 0; k < K; k++) {
			for (int t = 0; t < V; t++) {
				phi[k][t] = (nkt.get(k)[t] + beta) / (nk.get(k) + beta * V);
			}
		}
		for (int m = 0; m < M; m++) {
			for (int k = 0; k < K; k++) {
				theta[m][k] = (nmk[m].get(k) + alpha)
						/ (w[m].length + alpha * K);
			}
		}
	}
	
	
}
