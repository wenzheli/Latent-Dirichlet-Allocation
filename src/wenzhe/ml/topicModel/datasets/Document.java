/*
 * (C) Copyright 2013, wenzhe li (nadalwz1115@hotmail.com) 
 */

package wenzhe.ml.topicModel.datasets;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Document class contains all the words and frequency associated with them
 * Each document is a short text, which contains words are splitted by the deliminator. 
 * 
 * @author wenzhe  nadalwz1115@hotmail.com
 */
public class Document {
	
	/**  document identification number, which is default as 0 if is not used  */
	public int docID = 0;     
	
	/** <term, frequency> map. The term denotes the index of that term in vocabulary set,
	 *  and frequency denotes the number of times this term occured in this document. 
	 */
	public Map<Integer, Integer> termFreqMaps = new HashMap<Integer, Integer>();
	
	/** terms contained in this document, as in the order it is appeared  */
	public List<Integer> terms = new ArrayList<Integer>();
	
	/** total number of words contained in this document */
	public int totalTerms;
	
	/** total number of unique words contained in this document */
	public int uniqueTerms;
	
	/**
	 * Constructor for the document class. 
	 * The input is array which contains all the indexs of the terms. 
	 * @param terms
	 */
	public Document(int[] terms){
		totalTerms = terms.length;
		for (Integer termIdx: terms){
			if (!termFreqMaps.containsKey(termIdx))
				termFreqMaps.put(termIdx, 1);
			else{
				int currCount = termFreqMaps.get(termIdx);
				termFreqMaps.put(termIdx, currCount + 1);
			}
			
			this.terms.add(termIdx);
		}
		uniqueTerms = termFreqMaps.size();
		
	}
	
	/**
	 * Constructor for the document class. 
	 * The input is list, which contains all the indexs of the terms. 
	 * @param terms
	 */
	public Document(List<Integer> terms){
		this.terms = terms;
		totalTerms = terms.size();
		for (Integer termIdx: terms){
			if (!termFreqMaps.containsKey(termIdx))
				termFreqMaps.put(termIdx, 1);
			else{
				int currCount = termFreqMaps.get(termIdx);
				termFreqMaps.put(termIdx, currCount + 1);
			}
		}
		uniqueTerms = termFreqMaps.size();
	}
	
	/**
	 *  get the document id
	 * @return   document ID
	 */
	public int getDocID(){
		return this.docID;
	}
	
	/**
	 * get the term frequency map
	 * @return  <term, frequency> map
	 */
	public Map<Integer, Integer> getTermFreqMaps(){
		return this.termFreqMaps;
	}
	
	/**
	 * get the total number of terms in this document
	 * @return   total number of terms
	 */
	public int getTotalTerms(){
		return this.totalTerms;
	}
	
	/**
	 * get the total number of unique terms in this document
	 * @return   total number of unique terms.
	 */
	public int getUniqueTerms(){
		return this.uniqueTerms;
	}
	
	/**
	 * get the frequency for given a term
	 * @param term  the query term
	 * @return      number of times the term occur in this document. 
	 */
	public int getCount(int term){
		if (termFreqMaps.containsKey(term))
			return termFreqMaps.get(term);
		else
			return -1;
	}
	
	/**
	 * get the list of terms in order that is appeared in the document. 
	 * @return
	 */
	public List<Integer> getTerms(){
		return this.terms;
	}
	
	/**
	 * get the term indexed by variable idx
	 * @param idx   index
	 * @return      term
	 */
	public int getTerm(int idx){
		return this.terms.get(idx);
	}
	
}
