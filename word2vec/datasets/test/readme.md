####Test dataset folder
Many kinds of tests can be designed in order to assess the word2vector quality.

Here we provide few examples for:
- Questions test: The questions test is a simple check to ensure the words representation is well vectorized.
Each line contains four words: the second word is an image of the first one. The fourth is the image of the third. 
When the representation is well vectorized the fourth word should be predicted by the following test:

```model.wv.most_similar(positive=[word_1, word_2], negative=[word_3], topn=5)```.

- The distance check: this test check that similar words are closer than different words.
Each line contains three words: <negative word> <space> <positive word> <space> <query word>.
The distance between the word to the right and the second word has to be smaller than the distance between the word
 to the right and the word to the left.
 
 Many other types of checks can be designed and they are welcome.