{
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5-final"
  },
  "orig_nbformat": 2,
  "kernelspec": {
   "name": "python38564bit98ec1c6b27464ed6a09501a54863ac47",
   "display_name": "Python 3.8.5 64-bit"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2,
 "cells": [
  {
   "source": [
    "# Naive Bayes And Sentiment Classification - AI Project 03- Mohsen Amjadi - 810896043"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "Many language processing tasks involve classification. In this project we apply the naive\n",
    "Bayes algorithm to text categorization, the task is assigning a label or category to an entire text or document."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "sentiment analysis is the extraction of sentiment, the positive or negative orientation that a writer expresses toward some object. a comment on a product expresses the author's sentiment toward the product. one of the versions of sentiment analysis is a binary classification task, and the words of the review provide excellent cues. in this project we have recommended and not recommended classes for comments of the DigiKala products."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "Most cases of classification in language processing are done via supervised machine learning. In supervised learning, we have a data set of input observations(here comment_train), each associated with some correct output (a ‘supervision signal’). The goal of the algorithm is to learn how to map from a new observation (comment_test) to a correct output.\n",
    "Our goal is to learn a classifier that is capable of mapping from a new document d to its correct class c ∈ C. \n",
    "A probabilistic classifier additionally will tell us the probability of the observation being in the class.\n",
    "Generative classifiers like Naive Bayes build a model of how a class could generate some input data. Given an observation, they return the class most likely to have generated the observation."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "We use a text document as if it were a bag-of-words, that is, an unordered set of words with their position ignored, keeping only their frequency in the document. \n"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 252,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "6000"
      ]
     },
     "metadata": {},
     "execution_count": 252
    }
   ],
   "source": [
    "import csv\n",
    "import re\n",
    "\n",
    "RECOMMENDED = \"recommended\"\n",
    "NOT_RECOMMENDED = \"not_recommended\"\n",
    "not_recommended_count = 0\n",
    "recommended_count = 0\n",
    "not_recommended_all_words_count = 0\n",
    "recommended_all_words_count = 0\n",
    "words_set = set()\n",
    "data_set = list()\n",
    "with open(\"./CA3_dataset/comment_train.csv\", encoding=\"utf8\") as csvfile:\n",
    "            csvreader = csv.reader(csvfile, delimiter=\",\")\n",
    "            next(csvreader)\n",
    "            data_size = 0\n",
    "            for title, comment, recommend in csvreader:\n",
    "                comment = re.split(' |\\u200c', comment)\n",
    "                data_size += 1\n",
    "                data_set.append([comment, recommend])\n",
    "                \n",
    "data_size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 253,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "800"
      ]
     },
     "metadata": {},
     "execution_count": 253
    }
   ],
   "source": [
    "test_set = list()\n",
    "with open(\"./CA3_dataset/comment_test.csv\", encoding=\"utf8\") as csvfile:\n",
    "            csvreader = csv.reader(csvfile, delimiter=\",\")\n",
    "            next(csvreader)\n",
    "            test_size = 0\n",
    "            for title, comment, recommend in csvreader:\n",
    "                comment = re.split(' |\\u200c', comment)\n",
    "                test_size += 1\n",
    "                test_set.append([comment, recommend])\n",
    "                \n",
    "\n",
    "test_size"
   ]
  },
  {
   "source": [
    "Naive Bayes is a probabilistic classifier, meaning that for a document d, out of all classes c ∈ C the classifier returns the class cˆ which has the maximum posterior probability given the document."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "$$ \\hat c = \\underset{c \\in C}{argmax} \\text{ } P(c \\mid d) $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "so here the \n",
    "<b>Posterior $P(c|d)$  </b>Probability is Probability of a word belonging to our recommended or not recommended class given the word.\n",
    "with that we could set the (recommend OR not recommended) class for comments of a new set of words."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "The intuition of Bayesian classification is to use Bayes’ rule , that is presented below:"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "$$ P(x \\mid y) = \\frac{P(y \\mid x) \\, P(x)}{P(y)} $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "it gives us a way to break down any conditional probability P(x|y) into three other probabilities."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "so here we have :\n",
    "$$ \\hat c = \\underset{c \\in C}{argmax} \\text{ } P(c \\mid d) = {argmax} \\text{ } \\frac{P(d \\mid c) \\, P(c)}{P(d)} $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "We can conveniently simplify it by dropping the denominator P(d). This is possible because we will be computing the equation for each possible class. But P(d) doesn’t change for each class; we are always asking about the most likely class for the same document d, which must have the same probability P(d). Thus, we can choose the class that maximizes this simpler formula:\n",
    "$$ \\hat c = \\underset{c \\in C}{argmax} \\text{ } P(c \\mid d) = {argmax} \\text{ } {P(d \\mid c) \\, P(c)} $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "We thus compute the most probable class cˆ given some document d by choosing the class which has the highest product of two probabilities: the prior probability of the class P(c) and the likelihood of the document P(d|c):"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "likelihood : $$ {P(d \\mid c)} $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "Prior : $$ P(c) $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "Without loss of generalization, we can represent a document d as a set of features f1, f2,..., fn:\n",
    "$$ \\hat c = \\underset{c \\in C}{argmax} \\text{ } {P(f_1,f_2, ... ,f_n \\mid c) \\, P(c)} $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "this is still too hard to compute directly: without some simplifying assumptions, estimating the probability of every possible combination of\n",
    "features (for example, every possible set of words and positions) would require huge numbers of parameters and impossibly large training sets. Naive Bayes classifiers\n",
    "therefore make two simplifying assumptions:\n",
    "The first is the bag of words assumption , The second is commonly called the naive Bayes assumption: this is the conditional independence assumption that the probabilities P(fi|c) are independent given the class c and hence can be ‘naively’ multiplied as follows:"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "$$ P(f_1,f_2,...,f_n \\mid c) = P(f_1 \\mid c) P(f_2 \\mid c) ... P(f_n \\mid c) $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "The final equation for the class chosen by a naive Bayes classifier is thus:"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "N : Naive Bayes\n",
    "$$ c_N = \\underset{c \\in C}{argmax} \\text{ } P(c) \\underset{f \\in F} \\prod P(f \\mid c) $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "To apply the naive Bayes classifier to text, we need to consider word positions, by simply walking an index through every word position in the document:"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "$$ c_N = \\underset{c \\in C}{argmax} \\text{ } P(c) \\underset{i \\in positions} \\prod P(w_i \\mid c) $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 254,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "recommended_words_count = dict()\n",
    "not_recommended_words_count = dict()\n",
    "for data in data_set:\n",
    "    comment = data[0]\n",
    "    recommend = data[1]\n",
    "    if recommend == NOT_RECOMMENDED:\n",
    "        not_recommended_count += 1\n",
    "        for word in comment:\n",
    "            not_recommended_all_words_count += 1\n",
    "            words_set.add(word)\n",
    "            if word in not_recommended_words_count:\n",
    "                not_recommended_words_count[word] += 1\n",
    "            else:\n",
    "                not_recommended_words_count[word] = 1\n",
    "                \n",
    "    if recommend == RECOMMENDED:\n",
    "        recommended_count += 1\n",
    "        for word in comment:\n",
    "            recommended_all_words_count += 1\n",
    "            words_set.add(word)\n",
    "            if word in recommended_words_count:\n",
    "                recommended_words_count[word] += 1\n",
    "            else:\n",
    "                recommended_words_count[word] = 1\n",
    "\n",
    "#print(not_recommended_words_count)\n",
    "#print(recommended_words_count)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 255,
   "metadata": {},
   "outputs": [],
   "source": [
    "def calculate_recommended_p(comment):\n",
    "    #p = recommended_count/(recommended_count+not_recommended_count)  #p(C_k)\n",
    "    #p = recommended_all_words_count/(recommended_all_words_count+not_recommended_all_words_count)  #p(C_k)\n",
    "    p = len(not_recommended_words_count)/(len(recommended_words_count)+len(not_recommended_words_count))  #p(C_k)\n",
    "    #p = 0.5\n",
    "    #p = recommended_count/(recommended_count+not_recommended_count) #p(C_k)\n",
    "    for word in comment:\n",
    "        if word not in not_recommended_words_count and word not in recommended_words_count:\n",
    "            continue\n",
    "        if word in recommended_words_count:\n",
    "            p *= (recommended_words_count[word]/recommended_all_words_count)\n",
    "            #p += np.log(recommended_words_count[word]/len(recommended_words_count))\n",
    "        else:\n",
    "            p *= 0\n",
    "    return p\n",
    "            \n",
    "def calculate_not_recommended_p(comment):\n",
    "    p = not_recommended_count/(recommended_count+not_recommended_count)  #p(C_k)\n",
    "    #p = not_recommended_all_words_count/(recommended_all_words_count+not_recommended_all_words_count)  #p(C_k)\n",
    "    #p = len(not_recommended_words_count)/(len(recommended_words_count)+len(not_recommended_words_count))  #p(C_k)\n",
    "    #p = 0.5\n",
    "    #p = not_recommended_count/(recommended_count+not_recommended_count)  #p(C_k)\n",
    "    for word in comment:\n",
    "        if word not in not_recommended_words_count and word not in recommended_words_count:\n",
    "            continue\n",
    "        if word in not_recommended_words_count:\n",
    "            p *= (not_recommended_words_count[word]/not_recommended_all_words_count)\n",
    "            #p += np.log(not_recommended_words_count[word]/len(not_recommended_words_count))\n",
    "        else:\n",
    "            p *= 0\n",
    "    return p"
   ]
  },
  {
   "source": [
    "#  Evaluation: Precision, Recall, F-measure"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "To introduce the methods for evaluating text classification, let’s first consider some simple binary detection tasks. For example, in spam detection, our goal is to label every text as being in the spam category (“positive”) or not in the spam category (“negative”). For each item (email document) we therefore need to know whether\n",
    "our system called it spam or not. We also need to know whether the email is actually spam or not, i.e. the human-defined labels for each document that we are trying to\n",
    "match. We will refer to these human labels as the gold labels."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "Or imagine you’re the CEO of the Delicious Pie Company and you need to know what people are saying about your pies on social media, so you build a system that\n",
    "detects tweets concerning Delicious Pie. Here the positive class is tweets about Delicious Pie and the negative class is all other tweets"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "In both cases, we need a metric for knowing how well our spam detector (or pie-tweet-detector) is doing. To evaluate any system for detecting things, we start\n",
    "by building a contingency table. Each cell labels a set of possible outcomes. In the spam detection case, for example, true positives are documents that are indeed spam (indicated by human-created gold labels) and our system said they were spam. False negatives are documents that are indeed spam but our system labeled as non-spam."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "To the bottom right of the table is the equation for accuracy, which asks what percentage of all the observations (for the spam or pie examples that means all emails\n",
    "or tweets) our system labeled correctly. Although accuracy might seem a natural metric, we generally don’t use it. That’s because accuracy doesn’t work well when\n",
    "the classes are unbalanced (as indeed they are with spam, which is a large majority of email, or with tweets, which are mainly not about pie)."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "so we define accuracy as follows:\n",
    "$$  {Accuracy} \\text{ } = \\frac{{Correct Detected} \\text{ } }{{Total} \\text{ }} $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "To make this more explicit, imagine that we looked at a million tweets, and let’s say that only 100 of them are discussing their love (or hatred) for our pie,\n",
    "while the other 999,900 are tweets about something completely unrelated. Imagine a simple classifier that stupidly classified every tweet as “not about pie”. This classifier would have 999,900 true negatives and only 100 false negatives for an accuracy of 999,900/1,000,000 or 99.99%! Surely we should be happy with this classifier? But of course this fabulous ‘no pie’ classifier would be completely useless, since it wouldn’t find a single one of the customer comments we are looking for. In other words, accuracy is not a good metric when the goal is to discover something that is rare, or at least not completely balanced in frequency, which is a very common situation in the world."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "That’s why instead of accuracy we generally turn to two other metrics: precision and recall. Precision measures the percentage of the items that the system detected\n",
    "(i.e., the system labeled as positive) that are in fact positive (i.e., are positive according to the human gold labels). "
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "Precision is defined as : \n",
    "$$  {Precision} \\text{ } = \\frac{{Correct Detected Recommended} \\text{ } }{{All Detected Recommended (Including Wrong Ones)} \\text{ }} $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "Recall measures the percentage of items actually present in the input that were correctly identified by the system. "
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "Recall is defined as:\n",
    "$$  {Recall} \\text{ } = \\frac{{Correct Detected Recommended} \\text{ } }{{Total Recommended} \\text{ }} $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "Precision and recall will help solve the problem with the useless “nothing is pie” classifier. This classifier, despite having a fabulous accuracy of 99.99%, has\n",
    "a terrible recall of 0 (since there are no true positives, and 100 false negatives, the recall is 0/100). You should convince yourself that the precision at finding relevant tweets is equally problematic. Thus precision and recall, unlike accuracy, emphasize true positives: finding the things that we are supposed to be looking for."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "There are many ways to define a single metric that incorporates aspects of both F-measure precision and recall. The simplest of these combinations is the F-measure."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "$$  F_1 = 2 \\frac{{Precision*Recall} \\text{ } }{{Precision+Recall} \\text{ }} $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "F-measure comes from a weighted harmonic mean of precision and recall. The\n",
    "harmonic mean of a set of numbers is the reciprocal of the arithmetic mean of reciprocals:"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "$$  {HarmonicMean(a_1,a_2,a_3,...,a_n)} \\text{ } = \\frac{{n} \\text{ } }{\\frac{{1} \\text{ } }{{a_1} \\text{ }} + \\frac{{1} \\text{ } }{{a_2} \\text{ }} + \\frac{{1} \\text{ } }{{a_3} \\text{ }} + ... + \\frac{{1} \\text{ } }{{a_n} \\text{ }} } $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "Harmonic mean is used because it is a conservative metric; the harmonic mean of two values is closer to the minimum of the two values than the arithmetic mean is.\n",
    "Thus it weighs the lower of the two numbers more heavily."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 256,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Accuracy: 0.88\nRecall: 0.955\nPrecision: 0.8304347826086956\nF1: 0.8883720930232558\n"
     ]
    }
   ],
   "source": [
    "correct = 0\n",
    "correct_recommended = 0\n",
    "all_recommended = 0\n",
    "all_detected_recommended = 0\n",
    "for data in test_set:\n",
    "    comment = data[0]\n",
    "    recommend = data[1]\n",
    "    if recommend == RECOMMENDED:\n",
    "        all_recommended += 1\n",
    "        \n",
    "    if calculate_recommended_p(comment) >= calculate_not_recommended_p(comment):\n",
    "        all_detected_recommended += 1\n",
    "        if recommend == RECOMMENDED:\n",
    "            correct += 1\n",
    "            correct_recommended += 1\n",
    "    else:\n",
    "        if recommend == NOT_RECOMMENDED:\n",
    "            correct += 1\n",
    "        \n",
    "Accuracy = correct/len(test_set)\n",
    "Recall = correct_recommended/all_recommended\n",
    "Precision = correct_recommended/all_detected_recommended\n",
    "F1 = (2 * Precision * Recall) / (Precision + Recall)\n",
    "\n",
    "print(\"Accuracy:\", Accuracy)\n",
    "print(\"Recall:\", Recall)\n",
    "print(\"Precision:\", Precision)\n",
    "print(\"F1:\", F1)"
   ]
  },
  {
   "source": [
    "# Additive Smoothing (Laplace) smoothing"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "To learn the probability P(f_i\n",
    "|c), we’ll assume a feature is just the existence of a word\n",
    "in the document’s bag of words, and so we’ll want P(w_i\n",
    "|c), which we compute as\n",
    "the fraction of times the word w_i appears among all words in all documents of topic\n",
    "c. We first concatenate all documents with category c into one big “category c” text.\n",
    "Then we use the frequency of w_i\n",
    "in this concatenated document to give a maximum\n",
    "likelihood estimate of the probability:"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "$$ \\hat P(w_i \\mid c) = \\frac{{count} \\text{ } \\, (W_i,c)}{\\underset{w \\in V} \\sum_ ccount(w,c) } $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "Here the vocabulary V consists of the union of all the word types in all classes, not\n",
    "just the words in one class c."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "There is a problem, however, with maximum likelihood training. Imagine we\n",
    "are trying to estimate the likelihood of the word “fantastic” given class positive, but\n",
    "suppose there are no training documents that both contain the word “fantastic” and\n",
    "are classified as positive. Perhaps the word “fantastic” happens to occur in the class negative. In such a case the probability for this feature will be\n",
    "zero:"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "$$ \\hat P(fantastic \\mid positive) = \\frac{{count(fantastic , positive)} \\text{ } }{\\underset{w \\in V} \\sum_ ccount(w,positive) } = 0 $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "But since naive Bayes naively multiplies all the feature likelihoods together, zero\n",
    "probabilities in the likelihood term for any class will cause the probability of the\n",
    "class to be zero, no matter the other evidence!"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "The simplest solution is the add-one (Laplace) smoothing. "
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "$$ \\hat P(w_i \\mid c) = \\frac{{count(w_i,c) + 1} \\text{ }}{\\underset{w \\in V} \\sum_ (count(w,c) + 1) } = \\frac{{count(w_i,c) + 1} \\text{ }}{\\underset{w \\in V} (\\sum_ count(w,c)) +  |V| } $$"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "Note once again that it is crucial that the vocabulary V consists of the union of all the\n",
    "word types in all classes, not just the words in one class c."
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "What do we do about words that occur in our test data but are not in our vocabulary at all because they did not occur in any training document in any class? The solution for such unknown words is to ignore them—remove them from the test\n",
    "document and not include any probability for them at all"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "\n",
    "Finally, some systems choose to completely ignore another class of words: stop words, very frequent words like the and a. This can be done by sorting the vocabulary by frequency in the training set, and defining the top 10–100 vocabulary entries\n",
    "as stop words, or alternatively by using one of the many pre-defined stop word list\n",
    "available online. Then every instance of these stop words are simply removed from\n",
    "both training and test documents as if they had never occurred. In most text classification applications, however, using a stop word list doesn’t improve performance,\n",
    "and so it is more common to make use of the entire vocabulary and not use a stop\n",
    "word list"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 257,
   "metadata": {},
   "outputs": [],
   "source": [
    "def calculate_recommended_p(comment):\n",
    "    #p = recommended_count/(recommended_count+not_recommended_count)  #p(C_k)\n",
    "    #p = recommended_all_words_count/(recommended_all_words_count+not_recommended_all_words_count)  #p(C_k)\n",
    "    #p = len(recommended_words_count)/(len(recommended_words_count)+len(not_recommended_words_count))  #p(C_k)\n",
    "    p = 0.5\n",
    "    p = np.log(p)\n",
    "    for word in comment:\n",
    "        if word not in not_recommended_words_count and word not in recommended_words_count:\n",
    "            continue\n",
    "        if word in recommended_words_count:\n",
    "            word_count = recommended_words_count[word]\n",
    "        else:\n",
    "            word_count = 0\n",
    "        #p *= ((word_count+1)/(len(words_set) + recommended_all_words_count + 1 ))\n",
    "        p += np.log( ((word_count+1)/(len(words_set) + recommended_all_words_count + 1 )) )\n",
    "\n",
    "    return p\n",
    "            \n",
    "def calculate_not_recommended_p(text):\n",
    "    #p = not_recommended_count/(recommended_count+not_recommended_count)  #p(C_k)\n",
    "    #p = not_recommended_all_words_count/(recommended_all_words_count+not_recommended_all_words_count)  #p(C_k)\n",
    "    #p = len(not_recommended_words_count)/(len(recommended_words_count)+len(not_recommended_words_count))  #p(C_k)\n",
    "    p = 0.5\n",
    "    p = np.log(p)\n",
    "    for word in comment:\n",
    "        if word not in not_recommended_words_count and word not in recommended_words_count:\n",
    "            continue\n",
    "        if word in not_recommended_words_count:\n",
    "            word_count = not_recommended_words_count[word]\n",
    "        else:\n",
    "            word_count = 0\n",
    "            \n",
    "        #p *= ((word_count+1)/(len(words_set) + sadi_all_words_count + 1 ))\n",
    "        p += np.log( ((word_count+1)/(len(words_set) + not_recommended_all_words_count + 1 )) )\n",
    "\n",
    "    return p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 258,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Accuracy: 0.91875\nRecall: 0.945\nPrecision: 0.8978622327790974\nF1: 0.9208282582216808\n"
     ]
    }
   ],
   "source": [
    "correct = 0\n",
    "correct_recommended = 0\n",
    "all_recommended = 0\n",
    "all_detected_recommended = 0\n",
    "for data in test_set:\n",
    "    comment = data[0]\n",
    "    recommend = data[1]\n",
    "    if recommend == RECOMMENDED:\n",
    "        all_recommended += 1\n",
    "    if calculate_recommended_p(comment) >= calculate_not_recommended_p(comment):\n",
    "        all_detected_recommended += 1\n",
    "        if recommend == RECOMMENDED:\n",
    "            correct += 1\n",
    "            correct_recommended += 1\n",
    "\n",
    "    else:\n",
    "        if recommend == NOT_RECOMMENDED:\n",
    "            correct += 1\n",
    "        \n",
    "Accuracy = correct/len(test_set)\n",
    "Recall = correct_recommended/all_recommended\n",
    "Precision = correct_recommended/all_detected_recommended\n",
    "F1 = (2 * Precision * Recall) / (Precision + Recall)\n",
    "\n",
    "print(\"Accuracy:\", Accuracy)\n",
    "print(\"Recall:\", Recall)\n",
    "print(\"Precision:\", Precision)\n",
    "print(\"F1:\", F1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 259,
   "metadata": {},
   "outputs": [
    {
     "output_type": "display_data",
     "data": {
      "text/plain": "<Figure size 1440x648 with 1 Axes>",
      "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (https://matplotlib.org/) -->\n<svg height=\"535.43625pt\" version=\"1.1\" viewBox=\"0 0 1159.665625 535.43625\" width=\"1159.665625pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <metadata>\n  <rdf:RDF xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2020-11-19T20:13:21.349823</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.3.2, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linecap:butt;stroke-linejoin:round;}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 535.43625 \nL 1159.665625 535.43625 \nL 1159.665625 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 36.465625 511.558125 \nL 1152.465625 511.558125 \nL 1152.465625 22.318125 \nL 36.465625 22.318125 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"m8f5e3840b7\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"87.192898\" xlink:href=\"#m8f5e3840b7\" y=\"511.558125\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- simple -->\n      <g transform=\"translate(70.688991 526.156562)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 44.28125 53.078125 \nL 44.28125 44.578125 \nQ 40.484375 46.53125 36.375 47.5 \nQ 32.28125 48.484375 27.875 48.484375 \nQ 21.1875 48.484375 17.84375 46.4375 \nQ 14.5 44.390625 14.5 40.28125 \nQ 14.5 37.15625 16.890625 35.375 \nQ 19.28125 33.59375 26.515625 31.984375 \nL 29.59375 31.296875 \nQ 39.15625 29.25 43.1875 25.515625 \nQ 47.21875 21.78125 47.21875 15.09375 \nQ 47.21875 7.46875 41.1875 3.015625 \nQ 35.15625 -1.421875 24.609375 -1.421875 \nQ 20.21875 -1.421875 15.453125 -0.5625 \nQ 10.6875 0.296875 5.421875 2 \nL 5.421875 11.28125 \nQ 10.40625 8.6875 15.234375 7.390625 \nQ 20.0625 6.109375 24.8125 6.109375 \nQ 31.15625 6.109375 34.5625 8.28125 \nQ 37.984375 10.453125 37.984375 14.40625 \nQ 37.984375 18.0625 35.515625 20.015625 \nQ 33.0625 21.96875 24.703125 23.78125 \nL 21.578125 24.515625 \nQ 13.234375 26.265625 9.515625 29.90625 \nQ 5.8125 33.546875 5.8125 39.890625 \nQ 5.8125 47.609375 11.28125 51.796875 \nQ 16.75 56 26.8125 56 \nQ 31.78125 56 36.171875 55.265625 \nQ 40.578125 54.546875 44.28125 53.078125 \nz\n\" id=\"DejaVuSans-115\"/>\n        <path d=\"M 9.421875 54.6875 \nL 18.40625 54.6875 \nL 18.40625 0 \nL 9.421875 0 \nz\nM 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 64.59375 \nL 9.421875 64.59375 \nz\n\" id=\"DejaVuSans-105\"/>\n        <path d=\"M 52 44.1875 \nQ 55.375 50.25 60.0625 53.125 \nQ 64.75 56 71.09375 56 \nQ 79.640625 56 84.28125 50.015625 \nQ 88.921875 44.046875 88.921875 33.015625 \nL 88.921875 0 \nL 79.890625 0 \nL 79.890625 32.71875 \nQ 79.890625 40.578125 77.09375 44.375 \nQ 74.3125 48.1875 68.609375 48.1875 \nQ 61.625 48.1875 57.5625 43.546875 \nQ 53.515625 38.921875 53.515625 30.90625 \nL 53.515625 0 \nL 44.484375 0 \nL 44.484375 32.71875 \nQ 44.484375 40.625 41.703125 44.40625 \nQ 38.921875 48.1875 33.109375 48.1875 \nQ 26.21875 48.1875 22.15625 43.53125 \nQ 18.109375 38.875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.1875 51.21875 25.484375 53.609375 \nQ 29.78125 56 35.6875 56 \nQ 41.65625 56 45.828125 52.96875 \nQ 50 49.953125 52 44.1875 \nz\n\" id=\"DejaVuSans-109\"/>\n        <path d=\"M 18.109375 8.203125 \nL 18.109375 -20.796875 \nL 9.078125 -20.796875 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.390625 \nQ 20.953125 51.265625 25.265625 53.625 \nQ 29.59375 56 35.59375 56 \nQ 45.5625 56 51.78125 48.09375 \nQ 58.015625 40.1875 58.015625 27.296875 \nQ 58.015625 14.40625 51.78125 6.484375 \nQ 45.5625 -1.421875 35.59375 -1.421875 \nQ 29.59375 -1.421875 25.265625 0.953125 \nQ 20.953125 3.328125 18.109375 8.203125 \nz\nM 48.6875 27.296875 \nQ 48.6875 37.203125 44.609375 42.84375 \nQ 40.53125 48.484375 33.40625 48.484375 \nQ 26.265625 48.484375 22.1875 42.84375 \nQ 18.109375 37.203125 18.109375 27.296875 \nQ 18.109375 17.390625 22.1875 11.75 \nQ 26.265625 6.109375 33.40625 6.109375 \nQ 40.53125 6.109375 44.609375 11.75 \nQ 48.6875 17.390625 48.6875 27.296875 \nz\n\" id=\"DejaVuSans-112\"/>\n        <path d=\"M 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 0 \nL 9.421875 0 \nz\n\" id=\"DejaVuSans-108\"/>\n        <path d=\"M 56.203125 29.59375 \nL 56.203125 25.203125 \nL 14.890625 25.203125 \nQ 15.484375 15.921875 20.484375 11.0625 \nQ 25.484375 6.203125 34.421875 6.203125 \nQ 39.59375 6.203125 44.453125 7.46875 \nQ 49.3125 8.734375 54.109375 11.28125 \nL 54.109375 2.78125 \nQ 49.265625 0.734375 44.1875 -0.34375 \nQ 39.109375 -1.421875 33.890625 -1.421875 \nQ 20.796875 -1.421875 13.15625 6.1875 \nQ 5.515625 13.8125 5.515625 26.8125 \nQ 5.515625 40.234375 12.765625 48.109375 \nQ 20.015625 56 32.328125 56 \nQ 43.359375 56 49.78125 48.890625 \nQ 56.203125 41.796875 56.203125 29.59375 \nz\nM 47.21875 32.234375 \nQ 47.125 39.59375 43.09375 43.984375 \nQ 39.0625 48.390625 32.421875 48.390625 \nQ 24.90625 48.390625 20.390625 44.140625 \nQ 15.875 39.890625 15.1875 32.171875 \nz\n\" id=\"DejaVuSans-101\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-115\"/>\n       <use x=\"52.099609\" xlink:href=\"#DejaVuSans-105\"/>\n       <use x=\"79.882812\" xlink:href=\"#DejaVuSans-109\"/>\n       <use x=\"177.294922\" xlink:href=\"#DejaVuSans-112\"/>\n       <use x=\"240.771484\" xlink:href=\"#DejaVuSans-108\"/>\n       <use x=\"268.554688\" xlink:href=\"#DejaVuSans-101\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"594.465625\" xlink:href=\"#m8f5e3840b7\" y=\"511.558125\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- simple removing both 0 -->\n      <g transform=\"translate(535.010156 526.156562)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-32\"/>\n        <path d=\"M 41.109375 46.296875 \nQ 39.59375 47.171875 37.8125 47.578125 \nQ 36.03125 48 33.890625 48 \nQ 26.265625 48 22.1875 43.046875 \nQ 18.109375 38.09375 18.109375 28.8125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 20.953125 51.171875 25.484375 53.578125 \nQ 30.03125 56 36.53125 56 \nQ 37.453125 56 38.578125 55.875 \nQ 39.703125 55.765625 41.0625 55.515625 \nz\n\" id=\"DejaVuSans-114\"/>\n        <path d=\"M 30.609375 48.390625 \nQ 23.390625 48.390625 19.1875 42.75 \nQ 14.984375 37.109375 14.984375 27.296875 \nQ 14.984375 17.484375 19.15625 11.84375 \nQ 23.34375 6.203125 30.609375 6.203125 \nQ 37.796875 6.203125 41.984375 11.859375 \nQ 46.1875 17.53125 46.1875 27.296875 \nQ 46.1875 37.015625 41.984375 42.703125 \nQ 37.796875 48.390625 30.609375 48.390625 \nz\nM 30.609375 56 \nQ 42.328125 56 49.015625 48.375 \nQ 55.71875 40.765625 55.71875 27.296875 \nQ 55.71875 13.875 49.015625 6.21875 \nQ 42.328125 -1.421875 30.609375 -1.421875 \nQ 18.84375 -1.421875 12.171875 6.21875 \nQ 5.515625 13.875 5.515625 27.296875 \nQ 5.515625 40.765625 12.171875 48.375 \nQ 18.84375 56 30.609375 56 \nz\n\" id=\"DejaVuSans-111\"/>\n        <path d=\"M 2.984375 54.6875 \nL 12.5 54.6875 \nL 29.59375 8.796875 \nL 46.6875 54.6875 \nL 56.203125 54.6875 \nL 35.6875 0 \nL 23.484375 0 \nz\n\" id=\"DejaVuSans-118\"/>\n        <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-110\"/>\n        <path d=\"M 45.40625 27.984375 \nQ 45.40625 37.75 41.375 43.109375 \nQ 37.359375 48.484375 30.078125 48.484375 \nQ 22.859375 48.484375 18.828125 43.109375 \nQ 14.796875 37.75 14.796875 27.984375 \nQ 14.796875 18.265625 18.828125 12.890625 \nQ 22.859375 7.515625 30.078125 7.515625 \nQ 37.359375 7.515625 41.375 12.890625 \nQ 45.40625 18.265625 45.40625 27.984375 \nz\nM 54.390625 6.78125 \nQ 54.390625 -7.171875 48.1875 -13.984375 \nQ 42 -20.796875 29.203125 -20.796875 \nQ 24.46875 -20.796875 20.265625 -20.09375 \nQ 16.0625 -19.390625 12.109375 -17.921875 \nL 12.109375 -9.1875 \nQ 16.0625 -11.328125 19.921875 -12.34375 \nQ 23.78125 -13.375 27.78125 -13.375 \nQ 36.625 -13.375 41.015625 -8.765625 \nQ 45.40625 -4.15625 45.40625 5.171875 \nL 45.40625 9.625 \nQ 42.625 4.78125 38.28125 2.390625 \nQ 33.9375 0 27.875 0 \nQ 17.828125 0 11.671875 7.65625 \nQ 5.515625 15.328125 5.515625 27.984375 \nQ 5.515625 40.671875 11.671875 48.328125 \nQ 17.828125 56 27.875 56 \nQ 33.9375 56 38.28125 53.609375 \nQ 42.625 51.21875 45.40625 46.390625 \nL 45.40625 54.6875 \nL 54.390625 54.6875 \nz\n\" id=\"DejaVuSans-103\"/>\n        <path d=\"M 48.6875 27.296875 \nQ 48.6875 37.203125 44.609375 42.84375 \nQ 40.53125 48.484375 33.40625 48.484375 \nQ 26.265625 48.484375 22.1875 42.84375 \nQ 18.109375 37.203125 18.109375 27.296875 \nQ 18.109375 17.390625 22.1875 11.75 \nQ 26.265625 6.109375 33.40625 6.109375 \nQ 40.53125 6.109375 44.609375 11.75 \nQ 48.6875 17.390625 48.6875 27.296875 \nz\nM 18.109375 46.390625 \nQ 20.953125 51.265625 25.265625 53.625 \nQ 29.59375 56 35.59375 56 \nQ 45.5625 56 51.78125 48.09375 \nQ 58.015625 40.1875 58.015625 27.296875 \nQ 58.015625 14.40625 51.78125 6.484375 \nQ 45.5625 -1.421875 35.59375 -1.421875 \nQ 29.59375 -1.421875 25.265625 0.953125 \nQ 20.953125 3.328125 18.109375 8.203125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 75.984375 \nL 18.109375 75.984375 \nz\n\" id=\"DejaVuSans-98\"/>\n        <path d=\"M 18.3125 70.21875 \nL 18.3125 54.6875 \nL 36.8125 54.6875 \nL 36.8125 47.703125 \nL 18.3125 47.703125 \nL 18.3125 18.015625 \nQ 18.3125 11.328125 20.140625 9.421875 \nQ 21.96875 7.515625 27.59375 7.515625 \nL 36.8125 7.515625 \nL 36.8125 0 \nL 27.59375 0 \nQ 17.1875 0 13.234375 3.875 \nQ 9.28125 7.765625 9.28125 18.015625 \nL 9.28125 47.703125 \nL 2.6875 47.703125 \nL 2.6875 54.6875 \nL 9.28125 54.6875 \nL 9.28125 70.21875 \nz\n\" id=\"DejaVuSans-116\"/>\n        <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 75.984375 \nL 18.109375 75.984375 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-104\"/>\n        <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-48\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-115\"/>\n       <use x=\"52.099609\" xlink:href=\"#DejaVuSans-105\"/>\n       <use x=\"79.882812\" xlink:href=\"#DejaVuSans-109\"/>\n       <use x=\"177.294922\" xlink:href=\"#DejaVuSans-112\"/>\n       <use x=\"240.771484\" xlink:href=\"#DejaVuSans-108\"/>\n       <use x=\"268.554688\" xlink:href=\"#DejaVuSans-101\"/>\n       <use x=\"330.078125\" xlink:href=\"#DejaVuSans-32\"/>\n       <use x=\"361.865234\" xlink:href=\"#DejaVuSans-114\"/>\n       <use x=\"400.728516\" xlink:href=\"#DejaVuSans-101\"/>\n       <use x=\"462.251953\" xlink:href=\"#DejaVuSans-109\"/>\n       <use x=\"559.664062\" xlink:href=\"#DejaVuSans-111\"/>\n       <use x=\"620.845703\" xlink:href=\"#DejaVuSans-118\"/>\n       <use x=\"680.025391\" xlink:href=\"#DejaVuSans-105\"/>\n       <use x=\"707.808594\" xlink:href=\"#DejaVuSans-110\"/>\n       <use x=\"771.1875\" xlink:href=\"#DejaVuSans-103\"/>\n       <use x=\"834.664062\" xlink:href=\"#DejaVuSans-32\"/>\n       <use x=\"866.451172\" xlink:href=\"#DejaVuSans-98\"/>\n       <use x=\"929.927734\" xlink:href=\"#DejaVuSans-111\"/>\n       <use x=\"991.109375\" xlink:href=\"#DejaVuSans-116\"/>\n       <use x=\"1030.318359\" xlink:href=\"#DejaVuSans-104\"/>\n       <use x=\"1093.697266\" xlink:href=\"#DejaVuSans-32\"/>\n       <use x=\"1125.484375\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"1101.738352\" xlink:href=\"#m8f5e3840b7\" y=\"511.558125\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- Laplace Smoothing -->\n      <g transform=\"translate(1053.821946 526.156562)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 9.8125 72.90625 \nL 19.671875 72.90625 \nL 19.671875 8.296875 \nL 55.171875 8.296875 \nL 55.171875 0 \nL 9.8125 0 \nz\n\" id=\"DejaVuSans-76\"/>\n        <path d=\"M 34.28125 27.484375 \nQ 23.390625 27.484375 19.1875 25 \nQ 14.984375 22.515625 14.984375 16.5 \nQ 14.984375 11.71875 18.140625 8.90625 \nQ 21.296875 6.109375 26.703125 6.109375 \nQ 34.1875 6.109375 38.703125 11.40625 \nQ 43.21875 16.703125 43.21875 25.484375 \nL 43.21875 27.484375 \nz\nM 52.203125 31.203125 \nL 52.203125 0 \nL 43.21875 0 \nL 43.21875 8.296875 \nQ 40.140625 3.328125 35.546875 0.953125 \nQ 30.953125 -1.421875 24.3125 -1.421875 \nQ 15.921875 -1.421875 10.953125 3.296875 \nQ 6 8.015625 6 15.921875 \nQ 6 25.140625 12.171875 29.828125 \nQ 18.359375 34.515625 30.609375 34.515625 \nL 43.21875 34.515625 \nL 43.21875 35.40625 \nQ 43.21875 41.609375 39.140625 45 \nQ 35.0625 48.390625 27.6875 48.390625 \nQ 23 48.390625 18.546875 47.265625 \nQ 14.109375 46.140625 10.015625 43.890625 \nL 10.015625 52.203125 \nQ 14.9375 54.109375 19.578125 55.046875 \nQ 24.21875 56 28.609375 56 \nQ 40.484375 56 46.34375 49.84375 \nQ 52.203125 43.703125 52.203125 31.203125 \nz\n\" id=\"DejaVuSans-97\"/>\n        <path d=\"M 48.78125 52.59375 \nL 48.78125 44.1875 \nQ 44.96875 46.296875 41.140625 47.34375 \nQ 37.3125 48.390625 33.40625 48.390625 \nQ 24.65625 48.390625 19.8125 42.84375 \nQ 14.984375 37.3125 14.984375 27.296875 \nQ 14.984375 17.28125 19.8125 11.734375 \nQ 24.65625 6.203125 33.40625 6.203125 \nQ 37.3125 6.203125 41.140625 7.25 \nQ 44.96875 8.296875 48.78125 10.40625 \nL 48.78125 2.09375 \nQ 45.015625 0.34375 40.984375 -0.53125 \nQ 36.96875 -1.421875 32.421875 -1.421875 \nQ 20.0625 -1.421875 12.78125 6.34375 \nQ 5.515625 14.109375 5.515625 27.296875 \nQ 5.515625 40.671875 12.859375 48.328125 \nQ 20.21875 56 33.015625 56 \nQ 37.15625 56 41.109375 55.140625 \nQ 45.0625 54.296875 48.78125 52.59375 \nz\n\" id=\"DejaVuSans-99\"/>\n        <path d=\"M 53.515625 70.515625 \nL 53.515625 60.890625 \nQ 47.90625 63.578125 42.921875 64.890625 \nQ 37.9375 66.21875 33.296875 66.21875 \nQ 25.25 66.21875 20.875 63.09375 \nQ 16.5 59.96875 16.5 54.203125 \nQ 16.5 49.359375 19.40625 46.890625 \nQ 22.3125 44.4375 30.421875 42.921875 \nL 36.375 41.703125 \nQ 47.40625 39.59375 52.65625 34.296875 \nQ 57.90625 29 57.90625 20.125 \nQ 57.90625 9.515625 50.796875 4.046875 \nQ 43.703125 -1.421875 29.984375 -1.421875 \nQ 24.8125 -1.421875 18.96875 -0.25 \nQ 13.140625 0.921875 6.890625 3.21875 \nL 6.890625 13.375 \nQ 12.890625 10.015625 18.65625 8.296875 \nQ 24.421875 6.59375 29.984375 6.59375 \nQ 38.421875 6.59375 43.015625 9.90625 \nQ 47.609375 13.234375 47.609375 19.390625 \nQ 47.609375 24.75 44.3125 27.78125 \nQ 41.015625 30.8125 33.5 32.328125 \nL 27.484375 33.5 \nQ 16.453125 35.6875 11.515625 40.375 \nQ 6.59375 45.0625 6.59375 53.421875 \nQ 6.59375 63.09375 13.40625 68.65625 \nQ 20.21875 74.21875 32.171875 74.21875 \nQ 37.3125 74.21875 42.625 73.28125 \nQ 47.953125 72.359375 53.515625 70.515625 \nz\n\" id=\"DejaVuSans-83\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-76\"/>\n       <use x=\"55.712891\" xlink:href=\"#DejaVuSans-97\"/>\n       <use x=\"116.992188\" xlink:href=\"#DejaVuSans-112\"/>\n       <use x=\"180.46875\" xlink:href=\"#DejaVuSans-108\"/>\n       <use x=\"208.251953\" xlink:href=\"#DejaVuSans-97\"/>\n       <use x=\"269.53125\" xlink:href=\"#DejaVuSans-99\"/>\n       <use x=\"324.511719\" xlink:href=\"#DejaVuSans-101\"/>\n       <use x=\"386.035156\" xlink:href=\"#DejaVuSans-32\"/>\n       <use x=\"417.822266\" xlink:href=\"#DejaVuSans-83\"/>\n       <use x=\"481.298828\" xlink:href=\"#DejaVuSans-109\"/>\n       <use x=\"578.710938\" xlink:href=\"#DejaVuSans-111\"/>\n       <use x=\"639.892578\" xlink:href=\"#DejaVuSans-111\"/>\n       <use x=\"701.074219\" xlink:href=\"#DejaVuSans-116\"/>\n       <use x=\"740.283203\" xlink:href=\"#DejaVuSans-104\"/>\n       <use x=\"803.662109\" xlink:href=\"#DejaVuSans-105\"/>\n       <use x=\"831.445312\" xlink:href=\"#DejaVuSans-110\"/>\n       <use x=\"894.824219\" xlink:href=\"#DejaVuSans-103\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_4\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"m91966aa001\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"36.465625\" xlink:href=\"#m91966aa001\" y=\"452.256307\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 0.75 -->\n      <g transform=\"translate(7.2 456.055526)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 10.6875 12.40625 \nL 21 12.40625 \nL 21 0 \nL 10.6875 0 \nz\n\" id=\"DejaVuSans-46\"/>\n        <path d=\"M 8.203125 72.90625 \nL 55.078125 72.90625 \nL 55.078125 68.703125 \nL 28.609375 0 \nL 18.3125 0 \nL 43.21875 64.59375 \nL 8.203125 64.59375 \nz\n\" id=\"DejaVuSans-55\"/>\n        <path d=\"M 10.796875 72.90625 \nL 49.515625 72.90625 \nL 49.515625 64.59375 \nL 19.828125 64.59375 \nL 19.828125 46.734375 \nQ 21.96875 47.46875 24.109375 47.828125 \nQ 26.265625 48.1875 28.421875 48.1875 \nQ 40.625 48.1875 47.75 41.5 \nQ 54.890625 34.8125 54.890625 23.390625 \nQ 54.890625 11.625 47.5625 5.09375 \nQ 40.234375 -1.421875 26.90625 -1.421875 \nQ 22.3125 -1.421875 17.546875 -0.640625 \nQ 12.796875 0.140625 7.71875 1.703125 \nL 7.71875 11.625 \nQ 12.109375 9.234375 16.796875 8.0625 \nQ 21.484375 6.890625 26.703125 6.890625 \nQ 35.15625 6.890625 40.078125 11.328125 \nQ 45.015625 15.765625 45.015625 23.390625 \nQ 45.015625 31 40.078125 35.4375 \nQ 35.15625 39.890625 26.703125 39.890625 \nQ 22.75 39.890625 18.8125 39.015625 \nQ 14.890625 38.140625 10.796875 36.28125 \nz\n\" id=\"DejaVuSans-53\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-55\"/>\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-53\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_5\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"36.465625\" xlink:href=\"#m91966aa001\" y=\"359.597216\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 0.80 -->\n      <g transform=\"translate(7.2 363.396435)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 31.78125 34.625 \nQ 24.75 34.625 20.71875 30.859375 \nQ 16.703125 27.09375 16.703125 20.515625 \nQ 16.703125 13.921875 20.71875 10.15625 \nQ 24.75 6.390625 31.78125 6.390625 \nQ 38.8125 6.390625 42.859375 10.171875 \nQ 46.921875 13.96875 46.921875 20.515625 \nQ 46.921875 27.09375 42.890625 30.859375 \nQ 38.875 34.625 31.78125 34.625 \nz\nM 21.921875 38.8125 \nQ 15.578125 40.375 12.03125 44.71875 \nQ 8.5 49.078125 8.5 55.328125 \nQ 8.5 64.0625 14.71875 69.140625 \nQ 20.953125 74.21875 31.78125 74.21875 \nQ 42.671875 74.21875 48.875 69.140625 \nQ 55.078125 64.0625 55.078125 55.328125 \nQ 55.078125 49.078125 51.53125 44.71875 \nQ 48 40.375 41.703125 38.8125 \nQ 48.828125 37.15625 52.796875 32.3125 \nQ 56.78125 27.484375 56.78125 20.515625 \nQ 56.78125 9.90625 50.3125 4.234375 \nQ 43.84375 -1.421875 31.78125 -1.421875 \nQ 19.734375 -1.421875 13.25 4.234375 \nQ 6.78125 9.90625 6.78125 20.515625 \nQ 6.78125 27.484375 10.78125 32.3125 \nQ 14.796875 37.15625 21.921875 38.8125 \nz\nM 18.3125 54.390625 \nQ 18.3125 48.734375 21.84375 45.5625 \nQ 25.390625 42.390625 31.78125 42.390625 \nQ 38.140625 42.390625 41.71875 45.5625 \nQ 45.3125 48.734375 45.3125 54.390625 \nQ 45.3125 60.0625 41.71875 63.234375 \nQ 38.140625 66.40625 31.78125 66.40625 \nQ 25.390625 66.40625 21.84375 63.234375 \nQ 18.3125 60.0625 18.3125 54.390625 \nz\n\" id=\"DejaVuSans-56\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-56\"/>\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"36.465625\" xlink:href=\"#m91966aa001\" y=\"266.938125\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 0.85 -->\n      <g transform=\"translate(7.2 270.737344)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-56\"/>\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-53\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_7\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"36.465625\" xlink:href=\"#m91966aa001\" y=\"174.279034\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 0.90 -->\n      <g transform=\"translate(7.2 178.078253)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 10.984375 1.515625 \nL 10.984375 10.5 \nQ 14.703125 8.734375 18.5 7.8125 \nQ 22.3125 6.890625 25.984375 6.890625 \nQ 35.75 6.890625 40.890625 13.453125 \nQ 46.046875 20.015625 46.78125 33.40625 \nQ 43.953125 29.203125 39.59375 26.953125 \nQ 35.25 24.703125 29.984375 24.703125 \nQ 19.046875 24.703125 12.671875 31.3125 \nQ 6.296875 37.9375 6.296875 49.421875 \nQ 6.296875 60.640625 12.9375 67.421875 \nQ 19.578125 74.21875 30.609375 74.21875 \nQ 43.265625 74.21875 49.921875 64.515625 \nQ 56.59375 54.828125 56.59375 36.375 \nQ 56.59375 19.140625 48.40625 8.859375 \nQ 40.234375 -1.421875 26.421875 -1.421875 \nQ 22.703125 -1.421875 18.890625 -0.6875 \nQ 15.09375 0.046875 10.984375 1.515625 \nz\nM 30.609375 32.421875 \nQ 37.25 32.421875 41.125 36.953125 \nQ 45.015625 41.5 45.015625 49.421875 \nQ 45.015625 57.28125 41.125 61.84375 \nQ 37.25 66.40625 30.609375 66.40625 \nQ 23.96875 66.40625 20.09375 61.84375 \nQ 16.21875 57.28125 16.21875 49.421875 \nQ 16.21875 41.5 20.09375 36.953125 \nQ 23.96875 32.421875 30.609375 32.421875 \nz\n\" id=\"DejaVuSans-57\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-57\"/>\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"36.465625\" xlink:href=\"#m91966aa001\" y=\"81.619943\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 0.95 -->\n      <g transform=\"translate(7.2 85.419162)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-57\"/>\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-53\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_9\">\n    <path clip-path=\"url(#p0d06cb559d)\" d=\"M 87.192898 378.129034 \nL 594.465625 211.34267 \nL 1101.738352 155.747216 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:4;\"/>\n    <defs>\n     <path d=\"M 0 3 \nC 0.795609 3 1.55874 2.683901 2.12132 2.12132 \nC 2.683901 1.55874 3 0.795609 3 0 \nC 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132 \nC 1.55874 -2.683901 0.795609 -3 0 -3 \nC -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132 \nC -2.683901 -1.55874 -3 -0.795609 -3 0 \nC -3 0.795609 -2.683901 1.55874 -2.12132 2.12132 \nC -1.55874 2.683901 -0.795609 3 0 3 \nz\n\" id=\"m3266fa1b77\" style=\"stroke:#1f77b4;\"/>\n    </defs>\n    <g clip-path=\"url(#p0d06cb559d)\">\n     <use style=\"fill:#1f77b4;stroke:#1f77b4;\" x=\"87.192898\" xlink:href=\"#m3266fa1b77\" y=\"378.129034\"/>\n     <use style=\"fill:#1f77b4;stroke:#1f77b4;\" x=\"594.465625\" xlink:href=\"#m3266fa1b77\" y=\"211.34267\"/>\n     <use style=\"fill:#1f77b4;stroke:#1f77b4;\" x=\"1101.738352\" xlink:href=\"#m3266fa1b77\" y=\"155.747216\"/>\n    </g>\n   </g>\n   <g id=\"line2d_10\">\n    <path clip-path=\"url(#p0d06cb559d)\" d=\"M 87.192898 44.556307 \nL 594.465625 81.619943 \nL 1101.738352 100.151761 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-linecap:square;stroke-width:4;\"/>\n    <defs>\n     <path d=\"M 0 3 \nC 0.795609 3 1.55874 2.683901 2.12132 2.12132 \nC 2.683901 1.55874 3 0.795609 3 0 \nC 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132 \nC 1.55874 -2.683901 0.795609 -3 0 -3 \nC -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132 \nC -2.683901 -1.55874 -3 -0.795609 -3 0 \nC -3 0.795609 -2.683901 1.55874 -2.12132 2.12132 \nC -1.55874 2.683901 -0.795609 3 0 3 \nz\n\" id=\"mef20011876\" style=\"stroke:#ff7f0e;\"/>\n    </defs>\n    <g clip-path=\"url(#p0d06cb559d)\">\n     <use style=\"fill:#ff7f0e;stroke:#ff7f0e;\" x=\"87.192898\" xlink:href=\"#mef20011876\" y=\"44.556307\"/>\n     <use style=\"fill:#ff7f0e;stroke:#ff7f0e;\" x=\"594.465625\" xlink:href=\"#mef20011876\" y=\"81.619943\"/>\n     <use style=\"fill:#ff7f0e;stroke:#ff7f0e;\" x=\"1101.738352\" xlink:href=\"#mef20011876\" y=\"100.151761\"/>\n    </g>\n   </g>\n   <g id=\"line2d_11\">\n    <path clip-path=\"url(#p0d06cb559d)\" d=\"M 87.192898 489.319943 \nL 594.465625 304.001761 \nL 1101.738352 192.810852 \n\" style=\"fill:none;stroke:#2ca02c;stroke-linecap:square;stroke-width:4;\"/>\n    <defs>\n     <path d=\"M 0 3 \nC 0.795609 3 1.55874 2.683901 2.12132 2.12132 \nC 2.683901 1.55874 3 0.795609 3 0 \nC 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132 \nC 1.55874 -2.683901 0.795609 -3 0 -3 \nC -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132 \nC -2.683901 -1.55874 -3 -0.795609 -3 0 \nC -3 0.795609 -2.683901 1.55874 -2.12132 2.12132 \nC -1.55874 2.683901 -0.795609 3 0 3 \nz\n\" id=\"mc4ef5b00a8\" style=\"stroke:#2ca02c;\"/>\n    </defs>\n    <g clip-path=\"url(#p0d06cb559d)\">\n     <use style=\"fill:#2ca02c;stroke:#2ca02c;\" x=\"87.192898\" xlink:href=\"#mc4ef5b00a8\" y=\"489.319943\"/>\n     <use style=\"fill:#2ca02c;stroke:#2ca02c;\" x=\"594.465625\" xlink:href=\"#mc4ef5b00a8\" y=\"304.001761\"/>\n     <use style=\"fill:#2ca02c;stroke:#2ca02c;\" x=\"1101.738352\" xlink:href=\"#mc4ef5b00a8\" y=\"192.810852\"/>\n    </g>\n   </g>\n   <g id=\"line2d_12\">\n    <path clip-path=\"url(#p0d06cb559d)\" d=\"M 87.192898 248.406307 \nL 594.465625 211.34267 \nL 1101.738352 137.215398 \n\" style=\"fill:none;stroke:#d62728;stroke-linecap:square;stroke-width:4;\"/>\n    <defs>\n     <path d=\"M 0 3 \nC 0.795609 3 1.55874 2.683901 2.12132 2.12132 \nC 2.683901 1.55874 3 0.795609 3 0 \nC 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132 \nC 1.55874 -2.683901 0.795609 -3 0 -3 \nC -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132 \nC -2.683901 -1.55874 -3 -0.795609 -3 0 \nC -3 0.795609 -2.683901 1.55874 -2.12132 2.12132 \nC -1.55874 2.683901 -0.795609 3 0 3 \nz\n\" id=\"m8c8883d27e\" style=\"stroke:#d62728;\"/>\n    </defs>\n    <g clip-path=\"url(#p0d06cb559d)\">\n     <use style=\"fill:#d62728;stroke:#d62728;\" x=\"87.192898\" xlink:href=\"#m8c8883d27e\" y=\"248.406307\"/>\n     <use style=\"fill:#d62728;stroke:#d62728;\" x=\"594.465625\" xlink:href=\"#m8c8883d27e\" y=\"211.34267\"/>\n     <use style=\"fill:#d62728;stroke:#d62728;\" x=\"1101.738352\" xlink:href=\"#m8c8883d27e\" y=\"137.215398\"/>\n    </g>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 36.465625 511.558125 \nL 36.465625 22.318125 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 1152.465625 511.558125 \nL 1152.465625 22.318125 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 36.465625 511.558125 \nL 1152.465625 511.558125 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 36.465625 22.318125 \nL 1152.465625 22.318125 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"text_9\">\n    <!-- Accuracy Comparison -->\n    <g transform=\"translate(529.240938 16.318125)scale(0.12 -0.12)\">\n     <defs>\n      <path d=\"M 34.1875 63.1875 \nL 20.796875 26.90625 \nL 47.609375 26.90625 \nz\nM 28.609375 72.90625 \nL 39.796875 72.90625 \nL 67.578125 0 \nL 57.328125 0 \nL 50.6875 18.703125 \nL 17.828125 18.703125 \nL 11.1875 0 \nL 0.78125 0 \nz\n\" id=\"DejaVuSans-65\"/>\n      <path d=\"M 8.5 21.578125 \nL 8.5 54.6875 \nL 17.484375 54.6875 \nL 17.484375 21.921875 \nQ 17.484375 14.15625 20.5 10.265625 \nQ 23.53125 6.390625 29.59375 6.390625 \nQ 36.859375 6.390625 41.078125 11.03125 \nQ 45.3125 15.671875 45.3125 23.6875 \nL 45.3125 54.6875 \nL 54.296875 54.6875 \nL 54.296875 0 \nL 45.3125 0 \nL 45.3125 8.40625 \nQ 42.046875 3.421875 37.71875 1 \nQ 33.40625 -1.421875 27.6875 -1.421875 \nQ 18.265625 -1.421875 13.375 4.4375 \nQ 8.5 10.296875 8.5 21.578125 \nz\nM 31.109375 56 \nz\n\" id=\"DejaVuSans-117\"/>\n      <path d=\"M 32.171875 -5.078125 \nQ 28.375 -14.84375 24.75 -17.8125 \nQ 21.140625 -20.796875 15.09375 -20.796875 \nL 7.90625 -20.796875 \nL 7.90625 -13.28125 \nL 13.1875 -13.28125 \nQ 16.890625 -13.28125 18.9375 -11.515625 \nQ 21 -9.765625 23.484375 -3.21875 \nL 25.09375 0.875 \nL 2.984375 54.6875 \nL 12.5 54.6875 \nL 29.59375 11.921875 \nL 46.6875 54.6875 \nL 56.203125 54.6875 \nz\n\" id=\"DejaVuSans-121\"/>\n      <path d=\"M 64.40625 67.28125 \nL 64.40625 56.890625 \nQ 59.421875 61.53125 53.78125 63.8125 \nQ 48.140625 66.109375 41.796875 66.109375 \nQ 29.296875 66.109375 22.65625 58.46875 \nQ 16.015625 50.828125 16.015625 36.375 \nQ 16.015625 21.96875 22.65625 14.328125 \nQ 29.296875 6.6875 41.796875 6.6875 \nQ 48.140625 6.6875 53.78125 8.984375 \nQ 59.421875 11.28125 64.40625 15.921875 \nL 64.40625 5.609375 \nQ 59.234375 2.09375 53.4375 0.328125 \nQ 47.65625 -1.421875 41.21875 -1.421875 \nQ 24.65625 -1.421875 15.125 8.703125 \nQ 5.609375 18.84375 5.609375 36.375 \nQ 5.609375 53.953125 15.125 64.078125 \nQ 24.65625 74.21875 41.21875 74.21875 \nQ 47.75 74.21875 53.53125 72.484375 \nQ 59.328125 70.75 64.40625 67.28125 \nz\n\" id=\"DejaVuSans-67\"/>\n     </defs>\n     <use xlink:href=\"#DejaVuSans-65\"/>\n     <use x=\"66.658203\" xlink:href=\"#DejaVuSans-99\"/>\n     <use x=\"121.638672\" xlink:href=\"#DejaVuSans-99\"/>\n     <use x=\"176.619141\" xlink:href=\"#DejaVuSans-117\"/>\n     <use x=\"239.998047\" xlink:href=\"#DejaVuSans-114\"/>\n     <use x=\"281.111328\" xlink:href=\"#DejaVuSans-97\"/>\n     <use x=\"342.390625\" xlink:href=\"#DejaVuSans-99\"/>\n     <use x=\"397.371094\" xlink:href=\"#DejaVuSans-121\"/>\n     <use x=\"456.550781\" xlink:href=\"#DejaVuSans-32\"/>\n     <use x=\"488.337891\" xlink:href=\"#DejaVuSans-67\"/>\n     <use x=\"558.162109\" xlink:href=\"#DejaVuSans-111\"/>\n     <use x=\"619.34375\" xlink:href=\"#DejaVuSans-109\"/>\n     <use x=\"716.755859\" xlink:href=\"#DejaVuSans-112\"/>\n     <use x=\"780.232422\" xlink:href=\"#DejaVuSans-97\"/>\n     <use x=\"841.511719\" xlink:href=\"#DejaVuSans-114\"/>\n     <use x=\"882.625\" xlink:href=\"#DejaVuSans-105\"/>\n     <use x=\"910.408203\" xlink:href=\"#DejaVuSans-115\"/>\n     <use x=\"962.507812\" xlink:href=\"#DejaVuSans-111\"/>\n     <use x=\"1023.689453\" xlink:href=\"#DejaVuSans-110\"/>\n    </g>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 43.465625 89.030625 \nL 121.121875 89.030625 \nQ 123.121875 89.030625 123.121875 87.030625 \nL 123.121875 29.318125 \nQ 123.121875 27.318125 121.121875 27.318125 \nL 43.465625 27.318125 \nQ 41.465625 27.318125 41.465625 29.318125 \nL 41.465625 87.030625 \nQ 41.465625 89.030625 43.465625 89.030625 \nz\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\n    </g>\n    <g id=\"line2d_13\">\n     <path d=\"M 45.465625 35.416563 \nL 65.465625 35.416563 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:4;\"/>\n    </g>\n    <g id=\"line2d_14\">\n     <g>\n      <use style=\"fill:#1f77b4;stroke:#1f77b4;\" x=\"55.465625\" xlink:href=\"#m3266fa1b77\" y=\"35.416563\"/>\n     </g>\n    </g>\n    <g id=\"text_10\">\n     <!-- Accuracy -->\n     <g transform=\"translate(73.465625 38.916563)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-65\"/>\n      <use x=\"66.658203\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"121.638672\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"176.619141\" xlink:href=\"#DejaVuSans-117\"/>\n      <use x=\"239.998047\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"281.111328\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"342.390625\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"397.371094\" xlink:href=\"#DejaVuSans-121\"/>\n     </g>\n    </g>\n    <g id=\"line2d_15\">\n     <path d=\"M 45.465625 50.094688 \nL 65.465625 50.094688 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-linecap:square;stroke-width:4;\"/>\n    </g>\n    <g id=\"line2d_16\">\n     <g>\n      <use style=\"fill:#ff7f0e;stroke:#ff7f0e;\" x=\"55.465625\" xlink:href=\"#mef20011876\" y=\"50.094688\"/>\n     </g>\n    </g>\n    <g id=\"text_11\">\n     <!-- Recall -->\n     <g transform=\"translate(73.465625 53.594688)scale(0.1 -0.1)\">\n      <defs>\n       <path d=\"M 44.390625 34.1875 \nQ 47.5625 33.109375 50.5625 29.59375 \nQ 53.5625 26.078125 56.59375 19.921875 \nL 66.609375 0 \nL 56 0 \nL 46.6875 18.703125 \nQ 43.0625 26.03125 39.671875 28.421875 \nQ 36.28125 30.8125 30.421875 30.8125 \nL 19.671875 30.8125 \nL 19.671875 0 \nL 9.8125 0 \nL 9.8125 72.90625 \nL 32.078125 72.90625 \nQ 44.578125 72.90625 50.734375 67.671875 \nQ 56.890625 62.453125 56.890625 51.90625 \nQ 56.890625 45.015625 53.6875 40.46875 \nQ 50.484375 35.9375 44.390625 34.1875 \nz\nM 19.671875 64.796875 \nL 19.671875 38.921875 \nL 32.078125 38.921875 \nQ 39.203125 38.921875 42.84375 42.21875 \nQ 46.484375 45.515625 46.484375 51.90625 \nQ 46.484375 58.296875 42.84375 61.546875 \nQ 39.203125 64.796875 32.078125 64.796875 \nz\n\" id=\"DejaVuSans-82\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-82\"/>\n      <use x=\"64.982422\" xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"126.505859\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"181.486328\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"242.765625\" xlink:href=\"#DejaVuSans-108\"/>\n      <use x=\"270.548828\" xlink:href=\"#DejaVuSans-108\"/>\n     </g>\n    </g>\n    <g id=\"line2d_17\">\n     <path d=\"M 45.465625 64.772813 \nL 65.465625 64.772813 \n\" style=\"fill:none;stroke:#2ca02c;stroke-linecap:square;stroke-width:4;\"/>\n    </g>\n    <g id=\"line2d_18\">\n     <g>\n      <use style=\"fill:#2ca02c;stroke:#2ca02c;\" x=\"55.465625\" xlink:href=\"#mc4ef5b00a8\" y=\"64.772813\"/>\n     </g>\n    </g>\n    <g id=\"text_12\">\n     <!-- Precision -->\n     <g transform=\"translate(73.465625 68.272813)scale(0.1 -0.1)\">\n      <defs>\n       <path d=\"M 19.671875 64.796875 \nL 19.671875 37.40625 \nL 32.078125 37.40625 \nQ 38.96875 37.40625 42.71875 40.96875 \nQ 46.484375 44.53125 46.484375 51.125 \nQ 46.484375 57.671875 42.71875 61.234375 \nQ 38.96875 64.796875 32.078125 64.796875 \nz\nM 9.8125 72.90625 \nL 32.078125 72.90625 \nQ 44.34375 72.90625 50.609375 67.359375 \nQ 56.890625 61.8125 56.890625 51.125 \nQ 56.890625 40.328125 50.609375 34.8125 \nQ 44.34375 29.296875 32.078125 29.296875 \nL 19.671875 29.296875 \nL 19.671875 0 \nL 9.8125 0 \nz\n\" id=\"DejaVuSans-80\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-80\"/>\n      <use x=\"58.552734\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"97.416016\" xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"158.939453\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"213.919922\" xlink:href=\"#DejaVuSans-105\"/>\n      <use x=\"241.703125\" xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"293.802734\" xlink:href=\"#DejaVuSans-105\"/>\n      <use x=\"321.585938\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"382.767578\" xlink:href=\"#DejaVuSans-110\"/>\n     </g>\n    </g>\n    <g id=\"line2d_19\">\n     <path d=\"M 45.465625 79.450938 \nL 65.465625 79.450938 \n\" style=\"fill:none;stroke:#d62728;stroke-linecap:square;stroke-width:4;\"/>\n    </g>\n    <g id=\"line2d_20\">\n     <g>\n      <use style=\"fill:#d62728;stroke:#d62728;\" x=\"55.465625\" xlink:href=\"#m8c8883d27e\" y=\"79.450938\"/>\n     </g>\n    </g>\n    <g id=\"text_13\">\n     <!-- F1 -->\n     <g transform=\"translate(73.465625 82.950938)scale(0.1 -0.1)\">\n      <defs>\n       <path d=\"M 9.8125 72.90625 \nL 51.703125 72.90625 \nL 51.703125 64.59375 \nL 19.671875 64.59375 \nL 19.671875 43.109375 \nL 48.578125 43.109375 \nL 48.578125 34.8125 \nL 19.671875 34.8125 \nL 19.671875 0 \nL 9.8125 0 \nz\n\" id=\"DejaVuSans-70\"/>\n       <path d=\"M 12.40625 8.296875 \nL 28.515625 8.296875 \nL 28.515625 63.921875 \nL 10.984375 60.40625 \nL 10.984375 69.390625 \nL 28.421875 72.90625 \nL 38.28125 72.90625 \nL 38.28125 8.296875 \nL 54.390625 8.296875 \nL 54.390625 0 \nL 12.40625 0 \nz\n\" id=\"DejaVuSans-49\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-70\"/>\n      <use x=\"57.519531\" xlink:href=\"#DejaVuSans-49\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p0d06cb559d\">\n   <rect height=\"489.24\" width=\"1116\" x=\"36.465625\" y=\"22.318125\"/>\n  </clipPath>\n </defs>\n</svg>\n",
      "image/png": "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\n"
     },
     "metadata": {
      "needs_background": "light"
     }
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "accuracy = [0.79, 0.88, 0.91]\n",
    "recall = [0.97, 0.95, 0.94]\n",
    "precision = [0.73, 0.83, 0.89]\n",
    "F1 = [ 0.86 ,0.88 , 0.92]\n",
    "x = [\"simple\", \"simple removing both 0\", \"Laplace Smoothing\"]\n",
    "line_width = 4\n",
    "\n",
    "plt.figure(figsize=(20, 9))\n",
    "plt.plot(x, accuracy, '-o', label=\"Accuracy\", linewidth=line_width)\n",
    "plt.plot(x, recall, '-o', label=\"Recall\", linewidth=line_width)\n",
    "plt.plot(x, precision, '-o', label=\"Precision\", linewidth=line_width)\n",
    "plt.plot(x, F1, '-o', label=\"F1\", linewidth=line_width)\n",
    "\n",
    "plt.title(\"Accuracy Comparison\")\n",
    "#plt.xlabel(\"1: simple\\n2: simple removing both 0\\n3: Laplace Smoothing\")\n",
    "#plt.ylabel(\".\")\n",
    "plt.legend(loc=\"upper left\")\n",
    "plt.show()\n"
   ]
  }
 ]
}