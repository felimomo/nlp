import nltk
from gutenbergpy import textget # data cleaning for project gutenberg
from dataclasses import dataclass

# globals

BOOK_IDS = [
	42324, # frankenstein 
	18247, # the last man
	15238, # mathilda
	345, # dracula
]

# API

@dataclass
class gb_book:
	bk_id: int
	bk_content: bytes # gutenberg produces bytes as output

	def decode_text(self):
		return self.bk_content.decode()


def get_gb_book(book_id: int):
	raw_bk = textget.get_text_by_id(book_id)
	return textget.strip_headers(raw_bk)

# Corpus creation

library = [gb_book(bk_id, get_gb_book(bk_id)) for bk_id in BOOK_IDS]
corpus = " ".join([book.decode_text() for book in library])

