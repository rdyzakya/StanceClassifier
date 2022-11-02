import nltk

import datasets
import re

import argparse

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')