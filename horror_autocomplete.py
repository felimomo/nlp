import pprint
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.util import ngrams
import string
from gutenbergpy import textget # data cleaning for project gutenberg

# globals

BOOK_IDS = [
	42324, # frankenstein 
	18247, # the last man
	15238, # mathilda
	345, # dracula
	68283, # cthullu
	31469, # shunned house
	68236, # color out of space
	50133, # dunwich horror
	70912, # curse of yig
	70652, # mountains of madness
	25525, # works of poe, raven edition
	8486, # ghost stories antiquary
	5200, # metamorphosis
	43, # jekyll and clide
	389, # great god pan
	23172, # the damned thing
	10897, # the wendigo
	10007, # carmilla
	71168, # black canaan
	209, # turn of the screw
	71065, # the hyena
	71066, # dig me no grave
	71109, # black hound death
	71180, # grisly horror
	175, # phantom of the opera
	696, # castle of othranto
	14833, # varney the vampire
	11438, # the willows
	10662, # the night land
	5324, # book of werewolves
]

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.add("...")
UTF8_SC = {"\r", "\n", "\ufeff"} # UTF-8 special characters

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

	def n_sentences(self):
		return len(nltk.sent_tokenize(self.text))


# Fine-grained data pre-processing

def rem_punct(bk: gb_book):
	""" removes punctuation """
	bk.text = "".join([char for char in bk.text if char not in PUNCTUATION])
	return bk

def rem_SC(bk: gb_book):
	""" removes UTF-8 special characters """
	for sp_char in UTF8_SC:
		if sp_char in bk.text:
			bk.text = bk.text.replace(sp_char, "")
	return bk

def preprocess(bk: gb_book):
	""" removes special characters, upper cases and punctuation from book """
	bk = rem_punct(bk)
	bk = rem_SC(bk)
	bk.text = bk.text.lower()
	return bk

# N-gram analysis

def remove_SW_ngrams(ngram_list):     
	""" removes ngrams containing any stopwords """
	def gram_has_no_stop_word(gram):
		return not any([w in gram for w in STOP_WORDS])
	#
	return [ gram for gram in ngram_list if gram_has_no_stop_word(gram)]

def generate_ngrams(sentences, n: int, rm_sw: bool = True):
	"""
	Generates n-grams out of a sentence-tokenized text sentences.
	If rm_sw, then remove n-grams containing any stop-words
	"""

	tokenized_text =[
		word_tokenize(sentence)[:-1] 
		if sentence[-1]=='.'
		else word_tokenize(sentence)
		for sentence in sentences
	]
	ngram_list =  [
		pair for sequence in tokenized_text for pair in list(ngrams(sequence, n)) 
	] 

	if rm_sw:
		return remove_SW_ngrams(ngram_list)
	else: 
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

	# pprint.pprint({bk_id: bk.n_sentences() for bk_id, bk in library.items()})
	# -> dracula is by far the longest story and is introducing the statistics. 
	# need more data.


if __name__ == '__main__':
	main()




