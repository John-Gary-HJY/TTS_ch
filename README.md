# DuriTacoTTS

A text-to-speech synthesis project based on DurIAN/Tacotron architecture.

## Environment Setup

### Prerequisites
- CUDA 11.6 compatible GPU
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/ysanne617/DuriTacoTTS.git
cd DuriTacoTTS
```

2. Install uv (if not already installed):
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

3. Install Python 3.10 and create the environment:
```bash
uv python install 3.10
uv python pin 3.10
uv sync
```

## Usage

### Testing with Pretrained Model

1. Create a text file with one sentence per line and place it under `test_script/`:
```bash
echo "最近天氣又熱返，又落返雨" > test_script/test_demo.txt
echo "間房細細，但係好好住" >> test_script/test_demo.txt
```

2. Run the testing script:
```bash
sh testing.sh
```

### Training with New Speakers

1. **Prepare the speech corpus**

   Organize the collected recordings in the same format as `forced_alignment/speech_corpus/`. Each new speaker should have their own directory containing paired files:
   - `.wav` - audio recordings
   - `.lab` - corresponding Jyutping sequences

   Example structure:
   ```
   forced_alignment/speech_corpus/
   ├── speaker1/
   │   ├── utterance001.wav
   │   ├── utterance001.lab
   │   ├── utterance002.wav
   │   └── utterance002.lab
   └── speaker2/
       ├── ...
   ```
   **Note:** To convert Chinese text transcriptions into Jyutping sequences for `.lab` files, you can use the built-in text frontend:

   ```python
   from text.text_frontend import TextFrontend

   myfrontend = TextFrontend(lan='yue')
   jyutpings = myfrontend.text_to_jyutpings('最近天氣又熱返，又落返雨')
   print(jyutpings)
   ```

2. **Run Montreal Forced Aligner (MFA)**

   Generate phoneme-duration alignment information:

   ```bash
   cd forced_alignment
   sh mfa.sh
   ```

3. **Preprocess the data**

   Edit the preprocessing config file to specify:
   - Paths to the forced alignment results
   - Paths for the processed output files

   Then run:

   ```bash
   sh preprocess.sh
   ```

4. **Train the acoustic model**

   ```bash
   sh train_acoustic.sh
   ```

   Note: Add new speaker names to train_files/speaker_list.txt to include them in the training process.

5. **Train the duration predictor**

   ```bash
   sh train_duration_predictor.sh
   ```

6. **Run inference**

   Modify `testing.sh` to point to your new model checkpoint, then run:

   ```bash
   sh testing.sh
   ```

## Exercise
Use the recordings in `recordings/` (speaker: LvDou) to build a new TTS voice.