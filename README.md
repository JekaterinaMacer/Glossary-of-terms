# Glossary-of-terms
Glossary of common Machine Learning, Statistics and Data Science terms


#### algorithm
A series of repeatable steps for carrying out a certain type of task with data. As with data structures, people studying computer science learn about different algorithms and their suitability for various tasks. Specific data structures often play a role in how certain algorithms get implemented. 

#### artificial intelligence
Also, AI. The ability to have machines act with apparent intelligence, although varying definitions of “intelligence” lead to a range of meanings for the artificial variety. In AI’s early days in the 1960s, researchers sought general principles of intelligence to implement, often using symbolic logic to automate reasoning. As the cost of computing resources dropped, the focus moved more toward statistical analysis of large amounts of data to drive decision making that gives the appearance of intelligence. See also machine learning, data mining  

#### Bayes' Theorem
Also, Bayes' Rule. An equation for calculating the probability that something is true if something potentially related to it is true. If P(A) means “the probability that A is true” and P(A|B) means “the probability that A is true if B is true,” then Bayes' Theorem tells us that P(A|B) = (P(B|A)P(A)) / P(B). This is useful for working with false positives—for example, if x% of people have a disease, the test for it is correct y% of the time, and you test positive, Bayes' Theorem helps calculate the odds that you actually have the disease. The theorem also makes it easier to update a probability based on new data, which makes it valuable in the many applications where data continues to accumulate. Named for eighteenth-century English statistician and Presbyterian minister Thomas Bayes. See also Bayesian network, prior distribution  

#### Bayesian network
Also, Bayes net. “Bayesian networks are graphs that compactly represent the relationship between random variables for a given problem. These graphs aid in performing reasoning or decision making in the face of uncertainty. Such reasoning relies heavily on Bayes’ rule.”[bourg] These networks are usually represented as graphs in which the link between any two nodes is assigned a value representing the probabilistic relationship between those nodes. See also Bayes' Theorem, Markov Chain  

#### bias
In machine learning, “bias is a learner’s tendency to consistently learn the same wrong thing. Variance is the tendency to learn random things irrespective of the real signal.... It’s easy to avoid overfitting (variance) by falling into the opposite error of underfitting (bias). Simultaneously avoiding both requires learning a perfect classifier, and short of knowing it in advance there is no single technique that will always do best (no free lunch).”[domingos] See also variance, overfitting, classification  

#### Big Data
As this has become a popular marketing buzz phrase, definitions have proliferated, but in general, it refers to the ability to work with collections of data that had been impractical before because of their volume, velocity, and variety (“the three Vs”). A key driver of this new ability has been easier distribution of storage and processing across networks of inexpensive commodity hardware using technology such as Hadoop instead of requiring larger, more powerful individual computers. The work done with these large amounts of data often draws on data science skills.  

#### binomial distribution
A distribution of outcomes of independent events with two mutually exclusive possible outcomes, a fixed number of trials, and a constant probability of success. This is a discrete probability distribution, as opposed to continuous—for example, instead of graphing it with a line, you would use a histogram, because the potential outcomes are a discrete set of values. As the number of trials represented by a binomial distribution goes up, if the probability of success remains constant, the histogram bars will get thinner, and it will look more and more like a graph of normal distribution. See also probability distribution, discrete variable, histogram, normal distribution 

#### chi-square test
Chi (pronounced like “pie” but beginning with a “k”) is a Greek letter, and chi-square is “a statistical method used to test whether the classification of data can be ascribed to chance or to some underlying law.”[websters] The chi-square test “is an analysis technique used to estimate whether two variables in a cross tabulation are correlated.”[shin] A chi-square distribution varies from normal distribution based on the “degrees of freedom” used to calculate it. See also normal distribution and Wikipedia on the chi-squared test and on chi-squared distribution.

#### classification
The identification of which of two or more categories an item falls under; a classic machine learning task. Deciding whether an email message is spam or not classifies it among two categories, and analysis of data about movies might lead to classification of them among several genres. See also supervised learning, clustering  

#### clustering
Any unsupervised algorithm for dividing up data instances into groups—not a predetermined set of groups, which would make this classification, but groups identified by the execution of the algorithm because of similarities that it found among the instances. The center of each cluster is known by the excellent name “centroid.” See also classification, supervised learning, unsupervised learning, k-means clustering  

#### coefficient
“A number or algebraic symbol prefixed as a multiplier to a variable or unknown quantity (Ex.: x in x(y + z), 6 in 6ab”[websters] When graphing an equation such as y = 3x + 4, the coefficient of x determines the line's slope. Discussions of statistics often mention specific coefficients for specific tasks such as the correlation coefficient, Cramer’s coefficient, and the Gini coefficient. See also correlation  
