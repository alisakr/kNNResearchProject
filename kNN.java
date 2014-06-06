import java.io.*; 
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel; 
import java.util.*;
import java.lang.String;
import java.lang.Math;
import java.util.ArrayList;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;


/// This application uses an HDFS inverted index to classify documents using kNN
/// Usage:
/// hadoop jar simir.jar kNN InvertedIndexFileName Train_List Test_List K
/// -- "InvertedIndexFileName" is the name (including path) of the HDFS inverted file
///     (Make sure that you have all the three files: 
///                 + InvertedIndexFileName.lex: lexicon
///                 + InvertedIndexFileName.pos: posting
///                 + InvertedIndexFileName.dlen: doc length
/// -- "Train_List" is the name (including path) of the file which contains training documnet IDs and their class tag
///                 It has the following format (each document at a separate line)
///                 Tag1 DocumentID1
///                 Tag2 DocumentID2
/// -- "Test_list" is the name (including path) of the file which contains testing documents
///                  It has the following format (each testing document at a separate line)
///        DocumentID1 term1 term2 ... termN
///        DocumentID2 termN+1 termN+2 ....
/// -- "K" is the value for the parameter k in kNN algorithm 
	

/// This is an auxiliary class for sorting documents based on their scores. 
class ValueComparator implements Comparator { 
 
    Map base; 
    public ValueComparator(Map base) { 
	this.base = base; 
    } 
    
    public int compare(Object a, Object b) { 
	
	if((Double)base.get(a) < (Double)base.get(b)) { 
	    return 1; 
	} else if((Double)base.get(a) == (Double)base.get(b)) { 
	    return 0; 
	} else { 
	    return -1; 
	} 
    } 
}

    

/// This is the main class for kNN.
public class kNN {

    
    static double centerScore(int rawTF, int docFreq, int docCountTotal, int termCount, int totalTermCount, int docLength, double avgDocLength, int queryTF){
		double bm25Idf =0.5+ ((double)docCountTotal) - ((double) docFreq); 
		bm25Idf = bm25Idf /(0.5+docFreq);
		bm25Idf = Math.log(bm25Idf);
		double k1 = 1.6;
		double k2 = 1500.0;
		double b = 0.75;
		double queryTf = (double) queryTF;
		double bm25Tf = rawTF*(k1+1.0);
		bm25Tf = bm25Tf/(k1*((1.0-b)+ (b*((double) docLength)/avgDocLength)));
		double bm25query = (k2+1.0)*queryTf/(queryTf + k2);
		return bm25Idf*bm25Tf*bm25query;  

    }
    static int TOTALCLASS = 20;
    /// This function returns the weight of a matched query term for a document
    /// rawTF: raw count of the matched query term in the document
    /// docFreq: document frequency of the matched term (i.e.., total number of documents in the collection
    ///                 that contain the term
    /// docCountTotal: total number of documents in the collection
    /// termCount: the total count of the term in the whole collection
    /// totalTermCount: the sum of the total count of *all* the terms in the collection
    /// docLength: length of the document (number of words)
    /// avgDocLength: average document length in the collection
    /// param: a retrieval parameter that can be set to any value through the third argument when executing "Retrieval" 
    static double weight(int rawTF, int docFreq, int docCountTotal, int termCount,int totalTermCount, int docLength, double avgDocLength, double param, double avgCenterLen, boolean center, int centerLen) {
	double idf = Math.log((1.0+docCountTotal)/(0.5+docFreq));
        double bm25 = Math.log((0.5+docCountTotal-docFreq)/(0.5+docFreq)) ;
	double k = 2.0;
	double k2 = 1000.0;
	double b= 0.75;
	double tfNormalizer = 1-b + (b* ((double)docLength)/avgDocLength);
	double tfIdf = idf*((double)rawTF) /tfNormalizer ;
	//if(center){
	//	tfIdf = tfIdf * avgCenterLen/ (double) centerLen;
	//}
	return tfIdf; 
	// (ln ((N-df+0.5)/(df+0.5)))*(((k+1)*tf)/(k(1-b) + (b*dl/adl) + tf))
	// this is the raw TF-IDF weighting, which ignored a lot of information 

	// passed to this weighitng function; you may explore different ways of 
	// exploiting all the information to see if you can improve retrieval accuracy.
	// BM25, Dirichlet prior, and Pivoted length normalization are obvious choices, 
	// but you are encouraged to think about other choices. 
    }
    static HashMap<String, Double> termTFPair(String termFreqs){
	HashMap<String, Double> termsPair = new HashMap<String, Double>();
	StringTokenizer st = new StringTokenizer(termFreqs);
	String id = st.nextToken();
	return termsPair;
   }
    
    /// This is the core function of the kNN algorithm
    /// sortedAcc: the ranked document list of the current test document, and the closet document ranks the highest
    /// trainTag: this hashmap stores the document IDs and their category tags from the training data
    /// numK: this is the k value for kNN
    static int categorization(TreeMap<String, Double> sortedAcc, HashMap<String, Integer> trainTag, int numK, HashMap<String, Double> centerScores, double averageCenter, double beta, HashMap<String, Boolean> correctTag, HashMap<String, Integer> answers, String queryID, int[] centerLengths){

	/// initialize the vote counts for all the categories
	ArrayList<String> centersRanked = new ArrayList<String>();
	int [] counts = new int[TOTALCLASS];
	double[] queryWeights = new double[TOTALCLASS];
	double[] centerWeights = new double[TOTALCLASS];
	double[] weightedCounts = new double[TOTALCLASS];
	double avgCenter = 2031.23077;
	for(int j = 0; j < TOTALCLASS; j++){
	    counts[j] = 0;
	    centerWeights[j] = 0.0;
	    queryWeights[j] = 0.0;
	    weightedCounts[j] = 0.0;
	}
	int x = 0;
	HashMap<String, Double> adjCenters = new HashMap<String, Double>();	
	Integer  maxCenterCategory = 0;
	double maxCenter = 0.0;
	double maxNeighborScore = 0.0;
	double minCenter = 0.0;
	boolean hasFalseNeighbor = false;
	boolean minTagFalse = false;
	Set<Map.Entry<String,Double>> sortedSet = sortedAcc.entrySet();
	for (Map.Entry<String, Double> entry : sortedSet) {
		String key = entry.getKey();
		Double neighborScore = entry.getValue();
		Integer currTag = trainTag.get(key);
		if (currTag != null){
			x++;
			boolean accurateTag = correctTag.get(key);
			if(accurateTag == false){
				hasFalseNeighbor = true;
			}
			//System.err.println("category of  " + entry.getKey() + " is " + currTag);
			if(maxNeighborScore < neighborScore){
				maxNeighborScore = neighborScore;
			}
			if(centerScores.get(entry.getKey()) == null){
				System.err.println("centerScore is null");
				centerScores.put(entry.getKey(), 0.0);
				continue;
			}
			double b = 1.0;
			double centerAdjuster = 1-b +((1.0+b)*((double)centerLengths[currTag])/avgCenter);
			double centerAdjusted = centerScores.get(key)/centerAdjuster;
			adjCenters.put(key,centerAdjusted);
			if(maxCenter < adjCenters.get(key)){
				maxCenter = adjCenters.get(key);
				if(minCenter == 0.0){
					minCenter = maxCenter;
				}
				maxCenterCategory = currTag;
			}
			if(minCenter > centerScores.get(entry.getKey())){
				minCenter = centerScores.get(entry.getKey());
				if(accurateTag == false){
					minTagFalse = true;
				}
				else{
					minTagFalse = false;
				}
			}
	
		}
		if(x >= numK){
			//System.err.println("maxCenter is " + maxCenter);
			//System.err.println("maxNeighbor is " + maxNeighborScore);
			break;
		}		

	}
	if(hasFalseNeighbor){
		//System.err.println("query uses false neighbor, is a false neighbor the minimum center? " + minTagFalse); 

	}

	/// Look up from the top ranked results until we find numK labeled documents
	int i=0;
	//double queryWeights = new double[TOTALCLASS];
	boolean weightDisagreement = false;
	for (Map.Entry<String, Double> entry : sortedSet) { 
	    String key = entry.getKey(); 
	    Double neighborScore = entry.getValue();
	    /// Look up the tag of the document
	    Integer currTag = trainTag.get(key);
	    if(currTag != null){
		if(i == 0){
			if(currTag != maxCenterCategory){
				weightDisagreement = true;
				//System.err.println("max neighbor is of category " + currTag + " max center is of tag " + maxCenterCategory);
			}
			maxNeighborScore = neighborScore;
		}   
		counts[currTag] += 1;
 		i++;
		//double weightedCenter = (80.0*centerScores.get(key)/maxCenter);
		//double weightedQueryDist = (1.0-beta)*neighborScore;
		weightedCounts[currTag] += 1.0 ;
		if(centerScores.get(key) != null){
			//System.err.println( "center score is " + centerScores.get(key) + " and category is " + currTag);
			Double normalCenter = adjCenters.get(key)/maxCenter;
			//System.err.println("center normalized is " + normalCenter);
			Double normalNeighbor = neighborScore/maxNeighborScore ;
			//System.err.println("neighbor normalized is " + normalNeighbor);
			//System.err.println("neighbor tag is " + currTag + " centerScore is " + normalCenter);
			double neighborValue = 10.0*normalNeighbor;
			double centerValue = 10.0*normalCenter;			
			centerWeights[currTag] += centerValue;
			queryWeights[currTag] += neighborValue;
			weightedCounts[currTag] += ((1.0-beta)*neighborValue) + (beta*centerValue);

 		}
		else{
			System.err.println("Unreliable neighbor " + i  + "  has category " + currTag );
		}
		//#########################################################//
		// add statements here so that after the loop, counts would 
		// have the votes from numK nearest neighbors for each category 
		// Hint: how to update "counts" and what other variables need to update? 
		// 
		//#########################################################//

	    }else{
		/// if one top ranked document does not have a label from the training data
		/// then just skip this document
		;
	    }
	    if (i >= numK) {
		break;
	    }
	}

	/// find the category with the largest count
	int  maxClass = -1;
	double  maxCount = 0.0;
	int nClass = -1;
	double maxNcount = 0.0;
	int cClass = -1;
	double maxCcount = 0.0;
	Integer answer = answers.get(queryID);
	for(Integer j = 0; j < TOTALCLASS; j++){
	    if(centerWeights[j] > maxCcount){
	    	maxCcount = centerWeights[j];
		cClass = j;
	    }
	    if(queryWeights[j] > maxNcount){
		maxNcount = queryWeights[j];
		nClass = j;
	    }
	    if(weightedCounts[j] > maxCount){
		maxCount = weightedCounts[j];
		maxClass = j;
		//#########################################################//
		// add statements here so that after the loop, maxClass would
		// have the tag of the category which has the largest number of votes
		// Hint: two variables need to update here? 
		// 
		//#########################################################//
	    }
	}
	if(answer != maxClass){
		boolean centerOrNeighbor = false;
		double difference = 100.0*queryWeights[nClass] - 100.0*centerWeights[cClass]; 
		if(answer == nClass){
			centerOrNeighbor = true;
			//System.err.println(" wrong centerScore is " + centerWeights[answer] + " other neighbor score is " + queryWeights[maxClass]);
			//System.err.println("answer is wrong but neighbor query is correct, diff is "+ difference);
		}
		if(answer == cClass){
			difference = difference*(-1.0);
			centerOrNeighbor = true;
			//System.err.println("answer is wrong but center query is correct, diff is " + difference);
		}
		if(cClass != answer){
			System.err.println("on wrong category center length was " + centerLengths[cClass] + " in right answer it was " + centerLengths[answer]);
		}
	
			//System.err.println("answer is wrong, both weights are wrong");
		 	
	} 
	

	return maxClass;
	
    }


    public static void main (String [] args) throws IOException {
	

	/// This class defines the type Entry to pack all the information about a term stored in a lexicon entry. 
	class Entry {
	    public int df; // document frequency
	    public int count; // term count in the collection
	    public long pos; // start position of entries in the posting file
	    public int length; // span of postering entries 
	    Entry(int d, int c, long p, int l) {
		pos=p;
		length = l;
		df =d;
		count=c;
	    }
	}
	
	double retrievalModelParam = 0.5; // default retrieval parameter; this should be set to a meaningful value
	// for the retrieval model actually implemented. 

	// the following is standard HDFS setup 
	Configuration conf = new Configuration();
	FileSystem fs = FileSystem.get(conf);
	FSDataInputStream finlexicon=null, fintrain=null;
	FSDataInputStream  finposting=null, finposting2= null,findoclen=null, finquery=null; 
	FSDataInputStream answersFile = null;

	//Hash table for the lexicon:key is a term, value is an object of class Entry
	HashMap<String,Entry> lex= new HashMap<String,Entry>();

	// Hash table for the score accumulators: key is docID, value is score.
	HashMap<String,Double> acc = new HashMap<String,Double>();

	// Hash table for storing document length: key is docID, value is doc length
	HashMap<String,Integer> dlen = new HashMap<String,Integer>(); 

	HashMap<String,Boolean> correctTag = new HashMap<String, Boolean>();	

	// Hash table for storing the tags of training documents
	HashMap<String, Integer> trainTag = new HashMap<String, Integer>();
	HashMap<String, Double> centerScores = new HashMap<String, Double>();
	Entry termEntry = null;
	byte [] buffer = null; 
	String docID =null;
	int termFreq; 
	StringTokenizer st=null;
	String term =null;
	StringTokenizer tokenizeAnswers = null;
	int i; 
	double s; 

	int resultCount=1000; // this is the maximum number of results to return for each query
	int numK = 0; //this is the value of k for the kNN algorithm
	String cat11 = "the 572 a 258 to 255 in 229 of 217 and 170 said 151 hostag 119 i 104 that 98 wa 85 he 80 on 68 for 66 it 65 with 65 be 56 have 51 lebanon 50 at 50 by 47 report 45 american 43 releas 43 not 42 an 42 we 39 u 39 had 37 iran 37 held 37 offici 35 were 35 but 35 hi 35 ar 34 thei 33 been 31 beirut 30 who 30 from 29 ha 29 press 27 when 26 children 24 associ 24 after 23 polic 23 group 22 our 22 state 22 moslem 21 israel 21 will 21 western 20 their 20 kidnap 19 told 19 two 19 there 19 other 19 call 18 free 18 thi 18 shiit 18 ap 18 help 18 dai 18 school 17 would 17 fre 17 chief 17 apnr 17 hezbollah 16 if 16 perot 16 robertson 16 statem 16 nation 16 believ 16 foreign 16 washington 16 unit 15 kill 15 or 15 veri 15 iranian 15 offic 14 east 14 anderson 14 no 14 time 14 middl 14 todai 14 r 14 week 14 terri 14 made 14 new 13 prison 13 get 13 year 13 "; 
	
	if (args.length>=5) {
	    retrievalModelParam = Double.parseDouble(args[4]); // parse the provided parameter value if available.
	}

	String t=null;
	BufferedReader reader = null;
	try { 
	    // open the three files for the index
	    finposting = fs.open(new Path(args[0] + ".pos" ));
	    finlexicon = fs.open(new Path(args[0] + ".lex"));
	    findoclen = fs.open(new Path(args[0] + ".dlen"));
	    finposting2 = fs.open(new Path(args[0]+ ".pos"));
	    // open the training tag file
	    answersFile = fs.open(new Path(args[5]));
	    fintrain = fs.open(new Path(args[1]));

	    // open the query file
	    finquery = fs.open(new Path(args[2])); 

	    // read the value of K
	    numK = Integer.parseInt(args[3]);
	    double beta = Double.parseDouble(args[4]);	
	} catch (IOException ioe) {
	    ioe.printStackTrace();
	    System.out.println("file operation error: " + "args[0]="+ args[0] + ";args[1]="+args[1]); 
            System.exit(1);
	}

	// load the lexicon 
	int totalTermCount=0;
	while (finlexicon.available()!=0) {
	    term = finlexicon.readUTF(); 
	    int docFreq = finlexicon.readInt();
	    int termCount =finlexicon.readInt();
	    long  startPos = finlexicon.readLong();
	    int postingSpan = finlexicon.readInt();
	    lex.put(term,new Entry(docFreq,termCount,startPos, postingSpan)); 
	    totalTermCount += termCount;
	}	    
	finlexicon.close();
	
	// load doc length
	double avgDocLen =0;
	int totalDocCount=0;
	reader = new BufferedReader(new InputStreamReader(findoclen));
	while ((t=reader.readLine()) != null) {
	    st = new StringTokenizer(t);	
	    term = st.nextToken();
	    int docLen = Integer.parseInt(st.nextToken().trim());
	    dlen.put(term,docLen);

	    // we'll use this opportunity to compute the average doc length and the total number of documents in the collection
	    // note that it's better to precompute these values in the indexing stage and store them in a file
	    avgDocLen += docLen;  
	    totalDocCount++;
	}
	avgDocLen /= totalDocCount; 
	findoclen.close(); 
	int ctr = 0;
	int answersRead = 0;
	HashMap<String, Integer> answers = new HashMap<String, Integer>();
	reader = new BufferedReader(new InputStreamReader(answersFile));
	while((t=reader.readLine()) != null){
		st = new StringTokenizer(t);
		Integer tag = Integer.parseInt(st.nextToken());
		String doc = st.nextToken();
		answers.put(doc, tag);		
        }

	// load training tags
	reader = new BufferedReader(new InputStreamReader(fintrain));
	while ((t=reader.readLine()) != null) {
	    st = new StringTokenizer(t);
	    Integer currTag = Integer.parseInt(st.nextToken());
	    String doc = st.nextToken();
	    if(ctr == -1){
		Integer prev = currTag;
		Integer newTag = (int) (Math.random()*20);
		while(newTag == currTag){
			newTag = (int) (Math.random()*20);
		}
		currTag = newTag;
		correctTag.put(doc, false);
	    }
	    else{
		correctTag.put(doc,true); 
	    }
	    ctr++;
	    trainTag.put(doc, currTag);
	}

	// load training tags

    



        
        //int[] centerLengths = new int[TOTALCLASS] ;
	/*for(int y = 0; y < TOTALCLASS; y++){
		centerLengths[y] = 0;
	} */
	int[] centerLengths = {608, 2480, 0,1061, 2166, 0, 1973, 1943, 0, 1467,0, 1690, 0, 1673, 1403, 0, 1838, 0, 2483, 1442};

	boolean centerQuery = true;
	// process queries
	int z = 0; 
	reader = new BufferedReader(new InputStreamReader(finquery));
	while ((t=reader.readLine()) != null) {
	    Integer category = -1;
	    // each line has precisely one query: queryID term1 term 2.... 
	    st = new StringTokenizer(t); // A StringTokenizer allows us to decompose a string into space-separated tokens
	    String qid = st.nextToken(); // the first token should be the query ID
	    int centerLen = 0;
	    if(qid.contains("AP")){
		centerQuery = false;	
	    }
	    else{
		category = Integer.parseInt(qid);
		centerLen = centerLengths[category];
	    }
	    z++;	
	    System.err.println("step " + z + " Processing query:"+qid); 

	    acc.clear(); // clear the score accumulator to prepare for storing new scores for this query

	    int qlen=0; // counter for computing the query length

	    HashMap<String, Integer> qTermFreq = new HashMap<String, Integer>();

	    // trun the original query document into (term, freq) pairs
	    // this is to make the calculation faster
	    while (st.hasMoreTokens()) {
		term = st.nextToken();
		termEntry = lex.get(term);
	        //System.out.println("term is " + term);	
		if (termEntry != null) {
		    qlen++;
		    Integer currFreq = qTermFreq.get(term);
		    if(currFreq == null){
			qTermFreq.put(term, 1);
		    }else{
			qTermFreq.put(term, currFreq+1);
		    }
		}
	    }

	    double queryScore = 0.0;
	    int queryLength = 0;
	    // iterate over all the terms in the query document
	    for(Map.Entry<String, Integer> entry : qTermFreq.entrySet()) {
		term = entry.getKey();
		queryLength += entry.getValue();
		//System.out.println("term is " + term);
		termEntry = lex.get(term); // fetch the lexicon entry for this query term 
		boolean doEntry = true;
		if (termEntry != null) {
		    qlen++; 
		    int df = termEntry.df; 
		    // df tells us how many pairs (docID termCount) for this term we have in the posting file 
		    queryScore += weight(termEntry.count,df,totalDocCount,termEntry.count,totalTermCount,10,10.0, retrievalModelParam, 1837.35, false, 12); 
		    finposting.seek(termEntry.pos); // seek to the starting position of the posting entries for this term		
	
		    for (i=1; i<=df; i++) { // read in the df pairs 
			docID = finposting.readUTF().trim(); // read in a document ID
			termFreq = finposting.readInt();
			//System.out.println("doc is " + docID);
			if(centerQuery){
			 if (trainTag.get(docID) != category){
				doEntry = false;
				continue;
				}
			//System.out.println("docID is " + docID);
			}
			
			 // read in the term Count 
			int doclen = dlen.get(docID).intValue(); // fetch the document length for this doc 
	
			double tmpWeight = weight(termFreq,df,totalDocCount,termEntry.count,totalTermCount,doclen,avgDocLen, retrievalModelParam, 1837.35, centerQuery, centerLen );
			tmpWeight = tmpWeight * entry.getValue();
			// compute the weight of this matched term
			Double s1;
			if(centerQuery){
				s1 = centerScores.get(docID);
			}
			else{
				s1 = acc.get(docID); // get the current score for this docID in the accumulator if any
			}
			if (s1 != null) { 
			    // this means that the docID already has an entry in the accumulator, i.e., the docID already matched a previous query term
			   if(centerQuery){
				centerScores.put(docID, s1.doubleValue() + tmpWeight);
			   } 
			   else{
			   	acc.put(docID, s1.doubleValue() + tmpWeight);
			   }
			} else {
			    // otherwise, we need to add a score accumulator for this docID and set the score appropriately.
			   if(centerQuery){
				centerScores.put(docID, tmpWeight);
				//System.err.println("docID is " + docID +" and initial center is " + tmpWeight);
			   } 
			   else{
			   	acc.put(docID, tmpWeight);
			   }
			}
		    }
		    
		} else{
		    System.err.println("Skipping query term:"+term+ "(not in the collection)");
		}
	    }
	    

	    // At this point, we have iterated over all the query terms and updated the score accumulators appropriately
	    // so the score accumulators should have a sum of weights for all the matched query terms. 
	    // In some retrieval models, we may need to adjust this sum in some way, we can do it here
	    if(centerQuery){		
		centerLengths[category] = queryLength;
            }	  
	    if(centerQuery == false){	
	    // now we've finished scoring, and we'll sort the scores and output the top N results 
	    // to the standard output stream (System.out)
	    ValueComparator bvc =  new ValueComparator(acc); 
	    TreeMap<String,Double> sortedAcc = new TreeMap<String,Double>(bvc); 
 	    sortedAcc.putAll(acc); 
	    // call the core function of kNN algorithm
	    double beta = Double.parseDouble(args[4]);
	    int resTag = categorization(sortedAcc, trainTag, numK, centerScores, queryScore, beta, correctTag, answers, qid, centerLengths);

	    // print out the classification results
	    System.out.println(resTag + " " + qid);
	}
	}//end of a single query
	double sumCenters = 0.0;
	double numCenters = 0.0;
	for(int a= 0; a < TOTALCLASS; a++){
		if(centerLengths[a] != 0){
			sumCenters = sumCenters + (double) centerLengths[a];
			numCenters = numCenters + 1.0;
		}
		System.err.println("center " + a + " is " + centerLengths[a]);
	}
	double averageCenters = sumCenters/numCenters;
	System.err.println("the average center query length is " + averageCenters + "and the number of centers is " + numCenters); 
    }
	
}


