import sentencepiece as spm
# Use vocab size 30,000 as in BERT and ALBERT.
# To train faster, randomly sampling 10,000,000 sentences (lines).
spm.SentencePieceTrainer.train(input='spm_temp/cleaned_wiki103_bookcorpus_train.txt', model_prefix='spm_temp/wiki103_bookcorpus_spm', vocab_size=30000, input_sentence_size=10000000, shuffle_input_sentence=True)
