/*
 * (C) Copyright 2013, wenzhe li (nadalwz1115@hotmail.com) 
 */

package wenzhe.ml.topicModel.learning;

import java.util.List;

import wenzhe.ml.distributions.MultiNomialDist;
import wenzhe.ml.topicModel.datasets.Corpus;
import wenzhe.ml.topicModel.datasets.Document;

/**
 * The class represents the LDAModel, with some parameters. 
 * If you want to learn more about model, please refer to David Blei's paper (JMLR 2003)
 *
 * @author wenzhe
 *
 */
public class LDAModel {
	
	/** prior parameter for topic-document  */
	private double alpha;
	
	/** prior parameter for top-words */
	private double phi;
	
	/** hyperparameters for topic-word distributions,
	 *  this is  K by V matrix,  K: number of topics,  V:  number of terms in vocabulary 
	 */  
	private double[][] beta;
	
	/** hyperparameters for document-topic distributions, 
	 *  this is D by K matrix, where D:  number of documents, K: number of topics.
	 */
	private double[][] theta;
	
	/** total number of topics  */
	private int K; 
	
	/** vocabulary size  */
	private int V;
	
	/** total number of documents in the corpus */
	private int D;  
	
	/** latent variable, denote the word assignment to specific topic 
	 * the size of first dimension is D, where D is the number of documents,
	 * in this corpus, and second dimension varies by each document, which 
	 * is decided by the number of terms contained in each documents. 
	 */
	public int[][] z;
	
	
	/**
	 * D by K dimensional matrix, it stores the count of terms assigned to 
	 * topic k, for each document.  Each shows corresponds to one document. And each entry 
	 * DK_{i,j} means the number of terms assigned to topic j from document i. 
	 */
	private int[][] DK;
	
	/** D dimensional array, each entry DK_D means the number of terms in document i */
	private int[] DK_D;
	
	/**
	 * K by V dimensional matrix. Each entry KV_{i,j} means the the number of times term j assigned to 
	 * topic i. 
	 */
	private int[][] KV;
	
	/**
	 * K dimensional array, sum of the rows for KV. Each entry KV_K_{i} means the number of 
	 * terms assigned to topic i.  
	 */
	private int[] KV_K;

	/** number of iterations requred for gibbs sampler the default is value 100, 
	 *  you can change this value based on your data size, if your data size is bigger, you 
	 *  need to make it large to sufficiently converge 
	 */
	public int iterations = 100;
	
	/**
	 *  word index array. The first dimension is D, where D is the nubmer of documents 
	 *  in the corpus. 
	 */
	private int [][] doc;
	
	
	/**
	 * constructor. set the prior parameters for alpha and phi. 
	 * @param alpha   prior for doc-topic dist
	 * @param phi     prior for topic-word dist
	 * @param k       number of topics. 
	 */
    public LDAModel(double alpha, double phi, int k){
    	this.alpha = alpha;
    	this.phi = phi;
    	this.K = k;
    }
	
    /**
     * Initialize LDA model
     * assigning memory to the variable and initialize latent
     * variable Z. 
     * @param corpus
     */
    public void init(Corpus corpus){
    	D = corpus.getDocCount();   // set the number of documents. 
    	V = corpus.getTermCount();  // set the vocabulary size
    	
    	beta = new double[K][V];    
    	theta = new double[D][K];   
    	DK = new int[D][K];
    	KV = new int[K][V];
    	DK_D = new int[D];
    	KV_K = new int[K];
    	
    	List<Document> documents = corpus.getDocuments();
    	doc = new int[D][];
    	for (int i = 0 ; i < D; i++){
    		// construct each doc[i][] 
    		Document d = documents.get(i);
    		int count = d.getTotalTerms();
    		doc[i] = new int[count];
    		for (int j = 0; j < count; j++)
    			doc[i][j] = d.getTerm(j);
    	}
    	
    	// initialize the latent variable z
    	z= new int[D][];
    	for (int i = 0; i < D; i++){
    		Document d = documents.get(i);
    		int count = d.getTotalTerms();
    		z[i] = new int[count];
    		for (int j = 0; j < count; j++){
    			int randTopic = (int)(Math.random() * K);  // randomly sample from topic [1...K]. 
    			z[i][j] = randTopic;
    			DK[i][randTopic]++;
    			KV[randTopic][doc[i][j]]++;
    			KV_K[randTopic]++;
    		}
    		DK_D[i] = count;
    	}
    }
    
    /**
     * Run the collapsed gibbs sampling for given number
     * of iterations, specified by the model 
     */
    public void runCollapsedGibbs(){
    	for (int i = 0; i < iterations; i++){
    		for (int d = 0; d < D; d++){
    			int count = doc[d].length;
    			for (int w = 0; w < count; w++){
    				// run collapsed gibbs sampler, 
    				// sample from p(z_i|*)
    				int topic = sampleNewTopic(d,w);
    				z[d][w] = topic;
    			}
    		}
    	}
    	
    	updateParameters();
    }
    
    /**
     * update all the parameters. 
     */
    private void updateParameters(){
    	for(int k = 0; k < K; k++){
			for(int i = 0; i < V; i++){
				beta[k][i] = (KV[k][i] + phi) / (KV_K[k] + V * phi);
			}
		}
		
		for(int d = 0; d < D; d++){
			for(int k = 0; k < K; k++){
				theta[d][k] = (DK[d][k] + alpha) / (DK_D[d] + K * alpha);
			}
		}
    }
    
    /**
     * Sample the topic for word w, from document d
     * @param d    document d
     * @param w    word w from document d
     * @return     new topic for this word
     */
    private int sampleNewTopic(int d, int w){
    	int oldTopic = z[d][w];
    	DK[d][oldTopic]--;
    	KV[oldTopic][doc[d][w]]--;
    	DK_D[d]--;
    	KV_K[oldTopic]--;
    	
    	// compute p(z_i = j | *)
    	double[] p = new double[K];
    	for (int j = 0; j < K; j++){
    		p[j] = (alpha + DK[d][j]) / (K * alpha + DK_D[d]) * (phi + KV[j][doc[d][w]]) / (V * phi + KV_K[j]);
    	}
    			
    	// sample the topic topic from the distribution p[j].
    	MultiNomialDist dist = new MultiNomialDist(p);
    	int newTopic = dist.getSample();
    	
    	DK[d][newTopic]++;
    	KV[newTopic][doc[d][w]]++;
    	DK_D[d]++;
    	KV_K[newTopic]++;
    	
    	return newTopic;				
    }
    
    /**
     * get the beta, topic-word distributions.  
     * @return
     */
    public double[][] getBeta(){
    	return this.beta;
    }
    
    /**
     * get the words of this corpus
     * @return  words of this corpus
     */
    public int[][] getWords(){
    	return this.doc;
    }
    
    /**
     * get the theta, doc-topic distributions. 
     * @return
     */
    public double[][] getTheta(){
    	return this.theta;
    }
    
	/**
	 * set the prior parameter of alpha. 
	 * @param alpha
	 */
	public void setAlpah(double alpha){
		this.alpha = alpha;
	}
	
	/**
	 * set the prior paramter of phi
	 * @param phi
	 */
	public void setPhi(double phi){
		this.phi = phi;
	}
	
	/**
	 * get the topic number
	 * @return  K
	 */
	public int getTopicCount(){
		return this.K;
	}
	
	/**
	 * get the vocabulary size
	 * @return   vocabulary size
	 */
	public int getTermCount(){
		return this.V;
	}
	
	/**
	 * get the word assignment for document indexed by idx
	 * @param idx    the index of document
	 * @return       word assignments to the topics. 
	 */
	public int[] getWordAssignment(int idx){
		return z[idx];
	}
	
	/**
	 * get the number of documents in the corpus 
	 * @return   total number of documents. 
	 */
	public int getDocCount(){
		return this.D;
	}
	
	/**
	 * get the topic-word distributions. 
	 * @return   beta
	 */
	public double[][] getTopicWordDist(){
		return this.beta;
	}
	
	/**
	 * get the document-topic distributions. 
	 * @return   doc-topic distributions. 
	 */
	public double[][] getDocTopicDist(){
		return this.theta;
	}
	
	/**
	 * get the topic-word distribution, given a topic
	 * @param topic  topic index
	 * @return   topic-word distribution, given a topic
	 */
	public double[] getTopicWordDist(int topic){
		return this.beta[topic];
	}
	
	/**
	 * get the doc-topic distributions, given a document ID
	 * @param docID   document ID
	 * @return        topic distribution for this document. 
	 */
	public double[] getDocTopicDist(int docID){
		return this.theta[docID];
	}
}
