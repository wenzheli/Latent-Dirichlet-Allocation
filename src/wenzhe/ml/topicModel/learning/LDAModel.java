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
 * TODO will change maining convensions, right now ,just follow the notations that I used to 
 * derive the gibbs sampling.  Will update soon. 
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
	private int numTopics; 
	
	/** vocabulary size  */
	private int vocabSize;
	
	/** total number of documents in the corpus */
	private int numDocs;  
	
	/** latent variable, denote the word assignment to specific topic 
	 * the size of first dimension is D, where D is the number of documents,
	 * in this corpus, and second dimension varies by each document, which 
	 * is decided by the number of terms contained in each documents. 
	 */
	public int[][] z;
	
	
	/**
	 * D by K dimensional matrix, it stores the count of terms assigned to 
	 * topic k, for each document.  Each shows corresponds to one document. And each entry 
	 * docTopic_{i,j} means the number of terms assigned to topic j from document i. 
	 */
	private int[][] docTopic;
	
	/** D dimensional array, each entry of numTermsInDoc means the number of terms in document i */
	private int[] numTermsInDoc;
	
	/**
	 * K by V dimensional matrix. Each entry topicTerms{i,j} means the the number of times term j assigned to 
	 * topic i. 
	 */
	private int[][] topicTerms;
	
	/**
	 * K dimensional array, sum of the rows for topicTerms. Each entry termsCountInTopic{i} means the number of 
	 * terms assigned to topic i.  
	 */
	private int[] termsCountInTopic;

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
    public LDAModel(double alpha, double phi, int numTopics){
    	this.alpha = alpha;
    	this.phi = phi;
    	this.numTopics = numTopics;
    }
	
    /**
     * Initialize LDA model
     * assigning memory to the variable and initialize latent
     * variable Z. 
     * @param corpus
     */
    public void init(Corpus corpus){
        numDocs = corpus.getDocCount();   // set the number of documents. 
    	vocabSize = corpus.getTermCount();  // set the vocabulary size
    	
    	beta = new double[numTopics][vocabSize];    
    	theta = new double[numDocs][numTopics];   
    	docTopic = new int[numDocs][numTopics];
    	topicTerms = new int[numTopics][vocabSize];
    	numTermsInDoc = new int[numDocs];
    	termsCountInTopic = new int[numTopics];
    	
    	List<Document> documents = corpus.getDocuments();
    	doc = new int[numDocs][];
    	for (int i = 0 ; i < numDocs; i++){
    		// construct each doc[i][] 
    		Document d = documents.get(i);
    		int count = d.getTotalTerms();
    		doc[i] = new int[count];
    		for (int j = 0; j < count; j++)
    			doc[i][j] = d.getTerm(j);
    	}
    	
    	// initialize the latent variable z
    	z= new int[numDocs][];
    	for (int i = 0; i < numDocs; i++){
    		Document d = documents.get(i);
    		int count = d.getTotalTerms();
    		z[i] = new int[count];
    		for (int j = 0; j < count; j++){
    			int randTopic = (int)(Math.random() * numTopics);  // randomly sample from topic [1...K]. 
    			z[i][j] = randTopic;
    			docTopic[i][randTopic]++;
    			topicTerms[randTopic][doc[i][j]]++;
    			termsCountInTopic[randTopic]++;
    		}
    		numTermsInDoc[i] = count;
    	}
    }
    
    /**
     * Run the collapsed gibbs sampling for given number
     * of iterations, specified by the model 
     */
    public void runCollapsedGibbs(){
    	for (int i = 0; i < iterations; i++){
    		for (int d = 0; d < numDocs; d++){
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
    	for(int k = 0; k < numTopics; k++){
			for(int i = 0; i < vocabSize; i++){
				beta[k][i] = (topicTerms[k][i] + phi) / (termsCountInTopic[k] + vocabSize * phi);
			}
		}
		
		for(int d = 0; d < numDocs; d++){
			for(int k = 0; k < numTopics; k++){
				theta[d][k] = (docTopic[d][k] + alpha) / (numTermsInDoc[d] + numTopics * alpha);
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
    	docTopic[d][oldTopic]--;
    	topicTerms[oldTopic][doc[d][w]]--;
    	numTermsInDoc[d]--;
    	termsCountInTopic[oldTopic]--;
    	
    	// compute p(z_i = j | *)
    	double[] p = new double[numTopics];
    	for (int j = 0; j < numTopics; j++){
    		p[j] = (alpha + docTopic[d][j]) / (numTopics * alpha + numTermsInDoc[d]) * (phi + topicTerms[j][doc[d][w]]) / (vocabSize * phi + termsCountInTopic[j]);
    	}
    			
    	// sample the topic topic from the distribution p[j].
    	MultiNomialDist dist = new MultiNomialDist(p);
    	int newTopic = dist.getSample();
    	
    	docTopic[d][newTopic]++;
    	topicTerms[newTopic][doc[d][w]]++;
    	numTermsInDoc[d]++;
    	termsCountInTopic[newTopic]++;
    	
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
		return this.numTopics;
	}
	
	/**
	 * get the vocabulary size
	 * @return   vocabulary size
	 */
	public int getTermCount(){
		return this.vocabSize;
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
		return this.numDocs;
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
