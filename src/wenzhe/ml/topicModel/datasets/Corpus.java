/*
 * (C) Copyright 2013, wenzhe li (nadalwz1115@hotmail.com) 
 */

package wenzhe.ml.topicModel.datasets;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The class represents the whole corpus, which contains a list of documents. 
 * 
 * @author wenzhe   nadalwz1115@hotmail.com
 *
 */
public class Corpus {
	
	/** list of documents contained in this corpus  */
	List<Document> documents = new ArrayList<Document>();
	
	/** vocabulary map, each term maps to its index in the dictionary */
	Map<String, Integer>  vocabulary = new HashMap<String, Integer>();
	
	/** total number of terms in this vocabulary */
	int termCount;
	
	/** total number of documents in this corpus */
	int docCount;
	
	/**
	 * add document to the corupus
	 * @param doc
	 */
	public void addDocument(Document doc){
		documents.add(doc);
		docCount++;
	}
	
	/**
	 * get the vocabulary map. 
	 * @return
	 */
	public Map<String, Integer> getVocabulary(){
		return this.vocabulary;
	}
	/**
	 * set the vocabulary for this corupus
	 * @param vocab
	 */
	public void setVocabulary(Map<String, Integer> vocab){
		this.vocabulary = vocab;
		termCount = vocabulary.size();
	}
	
	/**
	 *  get the total number of documents in this corpus 
	 * @return   total number of documents
	 */
	public int getDocCount(){
		return this.docCount;
	}
	
	/**
	 * get the total number of unique terms in this corpus
	 * @return   total number of unique terms. 
	 */
	public int getTermCount(){
		return this.termCount;
	}
	
	/**
	 * get the list of documents in the corpus. 
	 * @return  list of documents. 
	 */
	public List<Document> getDocuments(){
		return this.documents;
	}
}
