import sentencepiece as spm
from prepro_utils import preprocess_text, encode_ids

# some code omitted here...
# initialize FLAGS


class XlnetTokenizer():
  def __init__(self, spiece_model_file, uncased=False):
    self.sp_model = spm.SentencePieceProcessor()
    self.sp_model.Load(spiece_model_file)
    self.uncased = uncased

  def tokenize(self, text):
    text = preprocess_text(text, lower=self.uncased)
    return encode_ids(self.sp_model, text)