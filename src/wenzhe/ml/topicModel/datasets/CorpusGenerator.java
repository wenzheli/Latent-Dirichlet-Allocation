/*
 * (C) Copyright 2013, wenzhe li (nadalwz1115@hotmail.com) 
 */

package wenzhe.ml.topicModel.datasets;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import wenzhe.ml.utils.Stopwords;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * Given a file path, generate the corpus object, which is used for
 * LDA and HDP model. 
 * 
 * @author wenzhe
 */
public class CorpusGenerator {
	
	private String filePath;   // path to the input file. 
	private int sampleSize = 1000;    // the number of documents we will sample
	
	/**
	 * constructor, create a new corpus. 
	 * @param filePath
	 */
	public CorpusGenerator(String filePath){
		this.filePath = filePath;
	}
	
	/**
	 * constructor, create a new corpus
	 * @param filePath    file path
	 * @param sampleSize  number of documents we will sample. 
	 */
	public CorpusGenerator(String filePath, int sampleSize){
		this.filePath = filePath;
		this.sampleSize = sampleSize;
	}
	
	/**
	 * get the file path for this corpus. 
	 * @return file path
	 */
	public String getFilePath(){
		return this.filePath;
	}
	
	/**
	 * create a new corpus. The input file is Json format. 
	 * All the reviews are included in one file separated by line break
	 * 
	 * @return   corpus instance
	 */
	public Corpus createCorpus(){
		BufferedReader br = null;
		String sCurrentLine;
		Gson gson = new Gson();
		JsonParser parser = new JsonParser();
		// vocabulary map . 
		Map<String, Integer> vob = new HashMap<String, Integer>();
		List<String> words = new ArrayList<String>();
		
		try{
			br = new BufferedReader(new FileReader(filePath));
			int idx = 0;
			int totals = 0;
			/* first path, iterate through all the reviews and construct the vocabulary list */
			while ((sCurrentLine = br.readLine()) != null) {
				if (totals > sampleSize)
					break;
				totals++;
				System.out.println(sCurrentLine);
			
				JsonObject ele = parser.parse(sCurrentLine).getAsJsonObject();
				String result = ele.get("text").getAsString();  // only get the text field, which is review text
				
				for (String str: result.split(" ")){
					str = str.trim();
				
					if (str.equals("")) continue;
					if (str.contains("\n")) continue;
					str = str.toLowerCase();
					if (Stopwords.isStopword(str)) continue;
				
					if (!vob.containsKey(str)){
						words.add(str);
						vob.put(str, idx);
						idx++;
					}
				}
			}
			
			/* second path, construct the document instance for each review, 
			 * also construct the corpus by adding all these documents into corpus
			 * instance
			 */
			br = new BufferedReader(new FileReader(filePath));
			totals = 0;
			Corpus corpus = new Corpus();
			while ((sCurrentLine = br.readLine()) != null) {
				if (totals > sampleSize)
					break;
				
				totals++;
				System.out.println(sCurrentLine);
				JsonObject ele = parser.parse(sCurrentLine).getAsJsonObject();
				String result = ele.get("text").getAsString();  // only get the text field, which is review text
				
				List<Integer> terms = new ArrayList<Integer>();
				for (String str: result.split(" ")){
					str = str.trim();
				
					if (str.equals("")) continue;
					if (str.contains("\n")) continue;
					str = str.toLowerCase();
					if (Stopwords.isStopword(str)) continue;
				
					// it is valid word, put it into list
					int index = vob.get(str);
					terms.add(index);
				}
				
				Document doc = new Document(terms);
				corpus.addDocument(doc);
			}
			
			corpus.setVocabulary(vob);
			
			return corpus;
			
		}catch(Exception e){
			System.out.println(e.getMessage());
			return null;
		}
	}
}
