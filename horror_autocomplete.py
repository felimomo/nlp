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
	# 42324, # frankenstein 
	# 18247, # the last man
	# 15238, # mathilda
	# 345, # dracula
	68283, # cthullu
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
	# 71168, # black canaan
	# 209, # turn of the screw
	# 71065, # the hyena
	# 71066, # dig me no grave
	# 71109, # black hound death
	# 71180, # grisly horror
	# 175, # phantom of the opera
	# 696, # castle of othranto
	# 14833, # varney the vampire
	# 11438, # the willows
	# 10662, # the night land
	# 5324, # book of werewolves
]

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update(
	{
		"...", "mr.", "dr.", "ms.", "mrs.", 
		"sir", "madam", "madamme", # generic
		"a.", "b.", "c.", "m." # for some reason, these were very common
	}
)
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

def rem_punct(s: str):
	""" removes punctuation """
	s = "".join([char for char in s if char not in PUNCTUATION])
	return s

def rem_SC(s: str):
	""" removes UTF-8 special characters """
	for sp_char in UTF8_SC:
		if sp_char in s:
			s = s.replace(sp_char, "")
	return s

def preprocess_str(s: str):
	""" removes special characters, upper cases and punctuation from string """
	s = rem_punct(s)
	s = rem_SC(s)
	s = s.lower()
	return s

def preprocess_book(bk: gb_book):
	""" removes special characters, upper cases and punctuation from book """
	bk.text = preprocess_str(bk.text)
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

def ngram_stats(sentences, n_list, verbose=False):
	freqs_dict = {}
	for n in n_list:
		ngram_list = generate_ngrams(sentences, n) 
		freq_n = nltk.FreqDist(ngram_list)
		freqs_dict[n] = freq_n
		if verbose:
			print(f"""
			Most common {n}-grams: {[(list(x),y) for (x,y) in freq_n.most_common(5)]}
			""")
	return freqs_dict


# PREDICTION

class ngram_predictor:

	def __init__(self, ngram_freqs: nltk.probability.FreqDist, n: int, vocabulary):
		self.ngram_freqs = ngram_freqs
		self.n = n
		self.vocabulary = vocabulary

	def str_to_predictor(self, in_text: str):
		preprocessed_text = preprocess_str(in_text)
		tokenized_text = word_tokenize(preprocessed_text)
		return [
			w for w in tokenized_text if w not in STOP_WORDS
		]

	def prune_branches(self, in_sequence, ngram_freqs: nltk.probability.FreqDist):
		""" 
		given a in_sequence word sequence, keep only ngram_freqs 
		which are compatible with it. 
		"""
		def item_compatible(in_sequence, gram):
			"""
			gram: a tuple of words
			predictor: tuple of words

			Compare whether the gram is a possible completion of the predictor
			sequence.
			"""
			sub_gram = gram[:-1]
			if len(in_sequence) >= len(sub_gram):
				return all(in_sequence[-len(sub_gram):] == sub_gram)
			#
			return all(in_sequence == sub_gram[:len(in_sequence)])
		
		return [ 
			(gram, freq)
			#
			for gram, freq in ngram_freqs.items() 
			#
			if item_compatible(in_sequence, gram)
		]

	def predictor_distribution(self, in_sequence):
		""" 
		from a set of frequencies, produces a distribution from which to
		sample n-grams.

		ngram_freqs_list: list of shape [(ngram, freq)]
		"""
		restr_ngram_freqs = self.prune_branches(
			in_sequence, self.ngram_freqs
		)

		if len(restr_ngram_freqs) > 0:
			total_N = sum(list(zip(*ngram_freqs_list))[1])
			return [ (gram[-1], freq/total_N) for (gram, freq) in ngram_freqs_list ]
		
		else:
			return [ (word, 1/len(self.vocabulary)) for word in vocabulary ]

				


def main():

	# Corpus creation and pre-processing
	#
	library = {bk_id: preprocess_book(gb_book(bk_id)) for bk_id in BOOK_IDS}

	corpus = " ".join(
		[ book.text for bk_id, book in library.items() ]
	)

	# N-grams
	#
	corpus_sentences = nltk.sent_tokenize(corpus)
	freq_d = ngram_stats(corpus_sentences, [1, 2, 3, 4])
	print(freq_d[4].items())

	# pprint.pprint({bk_id: bk.n_sentences() for bk_id, bk in library.items()})
	# -> dracula is by far the longest story and is introducing the statistics. 
	# need more data.


if __name__ == '__main__':
	main()




