mfa validate speech_corpus dictionary/lexicon.txt

mfa train --config_path ./mfa.config --clean speech_corpus dictionary/canton-lexicon.txt ./acoustic_model.zip aligned_speech_corpus
