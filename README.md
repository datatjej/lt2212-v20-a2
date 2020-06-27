# LT2212 V20 Assignment 2

Put any documentation here including any answers to the questions in the 
assignment on Canvas.

## Part 1
For tokenization I just split the data on white space and filtered out words containing only alphabetic charaters. I also applied NLTK's stop word list and limited the vocabulary to the words occuring more than 10 times in the corpus. This latter step decreased the vocabulary from 67k words to the more manageable 14,784. 

## Part 2
I used truncated SVD for the dimensionality reduction. 

## Part 3
-m 1 = KNeighborsClassifier(n_neighbors=3)<br> 
-m 2 = DecisionTreeClassifier()

## Part 4
<p>
<img src="results_table.PNG" alt="Table of results" width="80%" height="auto" border="10" /><br>
</p>

As the table above shows, the decision tree classifier is the clear winner when it comes to classifying the unreduced features, at 62% accuracy. The k-nearest neighbors classifier, set to the 3 nearest neighbors in this experiment, performs worse at ~42% accuracy, but improving somewhat by dimensionality reduction. It's interesting to see that the decision tree classifier drops so much in performance after dimensionality reduction, to between 24-28% accuracy. Would be nice to know why, but I couldn't find any easily googleable explanation for it. If time permitted it would also have been interesting to try out bigger windows for closest neighbours in the KNeighborsClassifier.  
