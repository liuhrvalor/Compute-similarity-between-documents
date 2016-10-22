/**
 * 
 */
package analyzer;

//import java.awt.List;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeMap;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;
import org.tartarus.snowball.ext.porterStemmer;

import structures.Post;
import structures.Token;

/**
 * @author Haoran Liu
 * @Date 10/25/2015
 * All other document are libraries 
 */

public class DocAnalyzer {

	// a list of stopwords
	HashSet<String> m_stopwords;

	HashMap<String, Double> m_idf;
	// store the loaded reviews in this arraylist for further processing
	ArrayList<Post> m_reviews;
	HashMap<String, Double> query_idf;

	TreeMap<String, String> treesort;
	// m_stats is for word count it is useful for my another program 
	// for validating Zipf's and computing IDF
	HashMap<String, Integer> m_stats;
	String total;

	Tokenizer tokenizer;
	PrintWriter pw;
	PrintWriter qw;
	PrintWriter mw;

	public DocAnalyzer() {
		treesort = new TreeMap<String, String>();
		m_reviews = new ArrayList<Post>();
		m_stats = new HashMap<String, Integer>();
		m_idf = new HashMap<String, Double>();
		m_stopwords = new HashSet<String>();
		query_idf=new HashMap<String, Double>();
		try {
			tokenizer = new TokenizerME(
					new TokenizerModel(
							new FileInputStream(
									"./data/Model/en-token.bin")));
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	// load a list of stopwords from file
	public void LoadStopwords(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) {
				// Stemming  and Normalization
				line = SnowballStemmingDemo(NormalizationDemo(line));
				if (!line.isEmpty())
					m_stopwords.add(line);
			}
			reader.close();
			System.out.format("Loading %d stopwords from %s\n",
					m_stopwords.size(), filename);
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
    
	
	// Preprocess the review data, tokenization, stemming and normalization
	// Store useful information in HashMap, such world count, reviews content
	public void analyzeDocumentDemo(JSONObject json) {
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			for (int i = 0; i < jarray.length(); i++) {
				Post review = new Post(jarray.getJSONObject(i));
				String content = review.getContent();


				HashSet<String> uniminiset = new HashSet<String>();
				HashSet<String> biminiset = new HashSet<String>();

				String[] unigram = tokenizer.tokenize(content);

				for (String token : unigram) {
					token = PorterStemmingDemo(SnowballStemmingDemo(NormalizationDemo(token)));

					uniminiset.add(token);
				}

				// count word frequency for unigram
				for (String token2 : uniminiset) {
					if (m_stopwords.contains(token2))
						continue;
					else {
						token2 = PorterStemmingDemo(SnowballStemmingDemo(NormalizationDemo(token2)));
						if (m_stats.containsKey(token2)) {
							m_stats.put(token2, m_stats.get(token2) + 1);
						} else {
							m_stats.put(token2, 1);
						}

					}

				}

				ArrayList<String> N_gram = new ArrayList<String>();

				for (String token : tokenizer.tokenize(content)) {

					token = PorterStemmingDemo(SnowballStemmingDemo(NormalizationDemo(token)));
					if (token.isEmpty()) {

						continue;
					} else {

						N_gram.add(token);
					}

				}

				String[] fine = new String[N_gram.size()];
                // In bigram, neither two words should occur in the stopwords
				for (int p = 0; p < N_gram.size() - 1; p++) {
					if (m_stopwords.contains(N_gram.get(p)))
						continue;
					if (m_stopwords.contains(N_gram.get(p + 1)))
						continue;

					fine[p] = N_gram.get(p) + "-" + N_gram.get(p + 1);

				}

				for (String str : fine) {

					biminiset.add(str);
				}
				
				// count word frequency for unigram
				for (String str2 : biminiset) {

					if (m_stats.containsKey(str2)) {
						m_stats.put(str2, m_stats.get(str2) + 1);
					} else {
						m_stats.put(str2, 1);
					}
				}
                
				// store review content on m_reviews so later on when want to use it 
				// we do not have to reread it from file
				m_reviews.add(review);
                
				//monitor the process of the program
				System.out.println("level" + m_reviews.size());

			}
		}

		catch (JSONException e) {
			e.printStackTrace();
		}

	}

	// function for loading a json file
	public JSONObject LoadJson(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;

			while ((line = reader.readLine()) != null) {
				buffer.append(line);
			}
			reader.close();

			return new JSONObject(buffer.toString());
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!", filename);
			e.printStackTrace();
			return null;
		} catch (JSONException e) {
			System.err.format("[Error]Failed to parse json file %s!", filename);
			e.printStackTrace();
			return null;
		}
	}

	//  load files in a directory recursively
	public void LoadDirectory(String folder, String suffix) {
		File dir = new File(folder);
		int size = m_reviews.size();
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix)) {
				analyzeDocumentDemo(LoadJson(f.getAbsolutePath()));
			} else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix);
		}

		// compute  Cosine similarity
		
		ComputeSimilarity();
    
		//This part of code is for my another program for validating Zipf's Law 
		// which requires to get word frequency and output it
		/*
		 * try { pw = new PrintWriter(new FileOutputStream(
		 * "C:/Users/Ryan/Desktop/information/IR task/IR task/data/Model/WC.txt"
		 * )); qw= new PrintWriter(new FileOutputStream(
		 * "C:/Users/Ryan/Desktop/information/IR task/IR task/data/Model/Fre.txt"
		 * )); mw= new PrintWriter(new FileOutputStream(
		 * "C:/Users/Ryan/Desktop/information/IR task/IR task/data/Model/Times.txt"
		 * ));
		 * 
		 * } catch(Exception e) {
		 * 
		 * }
		 * 
		 * pw.close();
		 */
		size = m_reviews.size() - size;
		System.out.println("Loading " + size + " review documents from "
				+ folder);
	}

	// Snowball stemmer
	public String SnowballStemmingDemo(String token) {
		SnowballStemmer stemmer = new englishStemmer();
		stemmer.setCurrent(token);
		if (stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}

	//Porter stemmer
	public String PorterStemmingDemo(String token) {
		porterStemmer stemmer = new porterStemmer();
		stemmer.setCurrent(token);
		if (stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}

	//  text normalization
	public String NormalizationDemo(String token) {

		token = token.replaceAll("\\p{Punct}+", "");
		token = token.replaceAll("\\W+", "");
		token = token.toLowerCase();
		if (token.matches("[+-]?[0-9]+(\\.[0-9]+)?")) {
			token = "NUM";
		}

		return token;
	}
   

	
	// Construct the vector space representations for these five reviews
	// the process is simliar with the first part of ComputeSimilarity() function
	public void analyzequery(JSONObject json) {
		try {
			
			JSONArray jarray = json.getJSONArray("Reviews");
			
			for (int i = 0; i < jarray.length(); i++) {
				
				Post review = new Post(jarray.getJSONObject(i));
				System.out.println("55");
				String content = review.getContent();
				
				
				

				HashMap<String, Integer> querytf = new HashMap<String, Integer>();

				HashSet<String> biminiset = new HashSet<String>();

				ArrayList uniquery = new ArrayList();

				for (String token : tokenizer.tokenize(content)) {
					if (m_stopwords.contains(token))
						continue;
					if (token.isEmpty())
						continue;

					uniquery.add(PorterStemmingDemo(SnowballStemmingDemo(NormalizationDemo(token))));

				}

				for (int k = 0; k < uniquery.size(); k++) {

					if (querytf.containsKey(uniquery.get(i).toString())) {
						querytf.put(uniquery.get(i).toString(),
								querytf.get(uniquery.get(i).toString()) + 1);
					} else {
						querytf.put(uniquery.get(i).toString(), 1);
					}

				}

				ArrayList<String> N_gram = new ArrayList<String>();

				for (String token : tokenizer.tokenize(content)) {

					token = PorterStemmingDemo(SnowballStemmingDemo(NormalizationDemo(token)));
					if (token.isEmpty()) {

						continue;
					} else {

						N_gram.add(token);
					}

				}

				String[] fine = new String[N_gram.size()];

				for (int p = 0; p < N_gram.size() - 1; p++) {
					if (m_stopwords.contains(N_gram.get(p)))
						continue;
					if (m_stopwords.contains(N_gram.get(p + 1)))
						continue;

					fine[p] = N_gram.get(p) + "-" + N_gram.get(p + 1);

				}

				
				for (String str2 : fine) {

					if (querytf.containsKey(str2)) {
						querytf.put(str2, querytf.get(str2) + 1);
					} else {
						querytf.put(str2, 1);
					}
				}

				for (String key : querytf.keySet()) {
					if (m_stats.containsKey(key)) {
						double df = (double) m_stats.get(key);

						double idf = (1 + Math.log(102201 / df));

						double tf = querytf.get(key);

						double result = tf * idf;

						query_idf.put(key, result);
					} else {
						query_idf.put(key, 0.0);
					}

				}
				
				
			}
			
			
		}
		catch (JSONException e) {
			e.printStackTrace();
		}
		
		}
	
	
	
   // Compute cosine similarity and output the 3 reviews with highest similarity 
	public void ComputeSimilarity() {
		
		//first construct the vector space representations for these five reviews
		// the our smaples vector, finally get the similarity metric 
		analyzequery(LoadJson("./data/samples/query.json"));
	
		
		HashMap<String, Double> Similarity = new HashMap<String, Double>();

		for (int i = 0; i < m_reviews.size(); i++) {
			String content = m_reviews.get(i).getContent();

			HashMap<String, Integer> conunttf = new HashMap<String, Integer>();

			HashSet<String> biminiset = new HashSet<String>();
            
			//danci means word unit: one or two words
			ArrayList danci = new ArrayList();

			for (String token : tokenizer.tokenize(content)) {
				if (m_stopwords.contains(token))
					continue;
				if (token.isEmpty())
					continue;

				danci.add(PorterStemmingDemo(SnowballStemmingDemo(NormalizationDemo(token))));

			}
            
			//get word count in a document
			for (int k = 0; k < danci.size(); k++) {

				if (conunttf.containsKey(danci.get(k).toString())) {
					conunttf.put(danci.get(k).toString(),
							conunttf.get(danci.get(k).toString()) + 1);
				} else {
					conunttf.put(danci.get(k).toString(), 1);
				}

			}

			ArrayList<String> N_gram = new ArrayList<String>();

			for (String token : tokenizer.tokenize(content)) {

				token = PorterStemmingDemo(SnowballStemmingDemo(NormalizationDemo(token)));
				if (token.isEmpty()) {

					continue;
				} else {

					N_gram.add(token);
				}

			}

			String[] fine = new String[N_gram.size()];
            
			//get rid of stopwords
			for (int p = 0; p < N_gram.size() - 1; p++) {
				if (m_stopwords.contains(N_gram.get(p)))
					continue;
				if (m_stopwords.contains(N_gram.get(p + 1)))
					continue;

				fine[p] = N_gram.get(p) + "-" + N_gram.get(p + 1);

			}

			
			for (String str2 : fine) {

				if (conunttf.containsKey(str2)) {
					conunttf.put(str2, conunttf.get(str2) + 1);
				} else {
					conunttf.put(str2, 1);
				}
			}
            
			//compute tf * idf for each document
			for (String key : conunttf.keySet()) {
				if (m_stats.containsKey(key)) {
					double df = (double) m_stats.get(key);

					double idf = (1 + Math.log(102201 / df));

					double tf = conunttf.get(key);

					double result = tf * idf;

					m_idf.put(key, result);
				} else {
					m_idf.put(key, 0.0);
				}

			}

			

			HashMap<String, Double> query = new HashMap<String, Double>();
			HashMap<String, Double> test = new HashMap<String, Double>();
            
			//If query contains this word, store it for future computation 
			for (Map.Entry<String, Double> entry : m_idf.entrySet()) {
				String key = entry.getKey();
				if (query_idf.containsKey(key)) {
					query.put(key, query_idf.get(key));
					test.put(key, m_idf.get(key));
				}
			}

			double dotProduct = 0.00;
			double magnitude1 = 0.0;
			double magnitude2 = 0.0;
			double magnitude3 = 0.0;
			double magnitude4 = 0.0;
			double cosineSimilarity = 0;
            
			
			//compute Compute similarity  between query document and each document in file
			for (String cal : test.keySet()) {
				dotProduct += query.get(cal) * test.get(cal); // a.b
				magnitude1 += Math.pow(query.get(cal), 2); // (a^2)
				magnitude2 += Math.pow(test.get(cal), 2); // (b^2)

				magnitude3 = Math.sqrt(magnitude1);// sqrt(a^2)
				magnitude4 = Math.sqrt(magnitude2);// sqrt(b^2)
			}

			if (magnitude3 != 0.0 | magnitude4 != 0.0)
				cosineSimilarity = dotProduct / (magnitude3 * magnitude4);

			else
				cosineSimilarity = 0;

			Similarity.put(content, cosineSimilarity);

					

		}

		// sort output to get 3 reviews with the highest similarity with query review
		List<Map.Entry<String, Double>> infoIds = new ArrayList<Map.Entry<String, Double>>(
				Similarity.entrySet());

		Collections.sort(infoIds, new Comparator<Map.Entry<String, Double>>() {
			public int compare(Map.Entry<String, Double> o1,
					Map.Entry<String, Double> o2) {
				return (int) (o1.getValue() - o2.getValue());
			}
		});

		for (int i = infoIds.size() - 1; i > infoIds.size() - 4; i--) {
			Entry<String, Double> ent = infoIds.get(i);
			System.out.println(ent.getValue()+"++"+ent.getKey()+"\n");

		}

	}

	public static void main(String[] args) {
		DocAnalyzer analyzer = new DocAnalyzer();
		
		//Load stopwords
		analyzer.LoadStopwords("./data/Model/english.stop.txt");
        
		// Load test document json file and make the computation to get the Compute similarity 
		analyzer.LoadDirectory("./data/test/yelp",".json");

	}

}
