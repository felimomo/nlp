import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import string
from gutenbergpy import textget # data cleaning for project gutenberg

# globals

BOOK_IDS = [
	42324, # frankenstein 
	# 18247, # the last man
	# 15238, # mathilda
	# 345, # dracula
	# 68283, # cthullu
	# 31469, # shunned house
	# 68236, # color out of space
	# 50133, # dunwich horror
	# 70912, # curse of yig
	# 70652, # mountains of madness
	# 25525, # works of poe, raven edition
	# 8486, # ghost stories antiquary
	# 5200, # metamorphosis
	# 43, # jekyll and clide
	# 389, # great god pan
	# 23172, # the damned thing
	# 10897, # the wendigo
	# 10007, # carmilla
]

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update({"\r", "\n", "\ufeff", "..."}) # UTF-8 special characters
print(STOP_WORDS)

PUNCTUATION = ( 
	string.punctuation.replace('.', '') # dont want periods here
	+'“'+'”'+'-'+'’'+'‘'+'—'
)

# gutenberg API

def get_gb_book(book_id: int):
	raw_bk = textget.get_text_by_id(book_id)
	return textget.strip_headers(raw_bk)

class gb_book:
	def __init__(self, bk_id: int):
		self.id = bk_id
		self.text = get_gb_book(self.id).decode() # bytes to string

	def to_nltk(self):
		return nltk.Text(self.text)

# Fine-grained data pre-processing

def rem_punct(bk: gb_book):
	""" removes punctuation """
	bk.text = "".join([char for char in bk.text if char not in PUNCTUATION])
	return bk

def rem_SW_SC(bk: gb_book):
	""" removes stop words and UTF-8 special characters """
	for word in STOP_WORDS:
		if word in bk.text:
			bk.text = bk.text.replace(word, "")
	return bk

### BUG: removing stop-words inside of other words as well! e.g. "forgiving" -> "giving" because "for" is erased. 
# TBD: deal with word-tokenization in nltk to handle this better
def preprocess(bk: gb_book):
	bk = rem_punct(bk)
	bk = rem_SW_SC(bk)
	bk.text = bk.text.lower()
	return bk

# N-gram analysis

def generate_ngrams(sentences, n):
	# sents = sentence-tokenized text (pretty much a list of sentences)
	# n = order of the ngrams (e.g. 2 for bigrams)

	tokenized_text =[
		word_tokenize(sentence.lower())[:-1] 
		if sentence[-1]=='.'
		else word_tokenize(sentence.lower())
		for sentence in sentences
	]

	ngram_list =  [
		pair for sequence in tokenized_text for pair in list(ngrams(sequence, n)) 
	] 

	return ngram_list

def ngram_stats(sentences, n_list):
	freqs_dict = {}
	for n in n_list:
		ngram_list = generate_ngrams(sentences, n) 
		freq_n = nltk.FreqDist(ngram_list)
		freqs_dict[n] = freq_n
		print(f"""
		Most common {n}-grams: {[(list(x),y) for (x,y) in freq_n.most_common(5)]}
		""")
	return freqs_dict


def main():

	# Corpus creation and pre-processing
	#
	library = {bk_id: preprocess(gb_book(bk_id)) for bk_id in BOOK_IDS}

	corpus = " ".join(
		[ book.text for bk_id, book in library.items() ]
	)

	# N-grams
	#
	corpus_sentences = nltk.sent_tokenize(corpus)
	freq_d = ngram_stats(corpus_sentences, [1, 2, 3, 4])



if __name__ == '__main__':
	main()




