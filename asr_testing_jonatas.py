# -*- coding: utf-8 -*-
"""asr_testing_jonatas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1opiN8HNGCKS8jXYtvxTAzeNiz6BzbM-u
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install datasets==1.18.3
# # !pip install transformers==4.17.0
# !pip install transformers==4.23.1
# !pip install jiwer
# !pip install huggingsound
 
# Commented out IPython magic to ensure Python compatibility.

# %cd /content

# %cd download

"""Data prep"""
import re
from datasets import load_dataset, load_metric, DatasetDict, Dataset, Audio
from huggingsound import SpeechRecognitionModel
import pandas as pd
import librosa
from tqdm import tqdm



def remove_special_characters(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).upper() + " "
    return batch

def prep_asr_testing(torgo_dataset):
    """
    
    """
    references = []
    for i in range(torgo_dataset.num_rows):
        row = {"path": torgo_dataset[i]["audio"]["path"],
        "transcription": torgo_dataset[i]["text"].lower()}
        references.append(row)
    return references


def prep_csv(file_path, min_length = 0, max_length = 0):
    """
    Preps the CSV with all of the data to ensure a max length and min length.
    Also writes csv to correct file name
    """
    df = pd.read_csv(file_path)

    duration = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        df.at[index, "audio"] = df.at[index, "audio"][1:]
        y, sr = librosa.load(df.at[index, "audio"], sr = 16000)
        duration.append(librosa.get_duration(y=y, sr=sr))

    df["duration"] = duration

    if max_length > 0 and min_length > 0:
         df = df[df["duration"].between(min_length, max_length)]
         df.to_csv("output.csv")
    elif max_length > 0:
         df = df[df["duration"] < max_length]
         df.to_csv("output.csv")
    elif min_length > 0:
        df = df[df["duration"] > min_length]
        df.to_csv("output.csv")
        

    
      

def main():
        
  
    speaker = 'M05'

  
    model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    # model = SpeechRecognitionModel("jeapaul/wav2vec2-large-xlsr-53-torgo-demo-M03-nolm")
    # model = SpeechRecognitionModel("yip-i/torgo_xlsr_finetune-" + speaker + "-2")
    

    data = load_dataset('csv', data_files='output.csv')
    data = data.cast_column("audio", Audio(sampling_rate=16_000))
    timit = data['train'].filter(lambda x: x == speaker, input_columns=['speaker_id'])

    timit[0]
    timit = timit.map(remove_special_characters)
    

    audio_paths = ["content/downloads/Torgo/F01/Session1/wav_arrayMic/0008.wav", "content/downloads/Torgo/F01/Session1/wav_arrayMic/0009.wav"]

    transcriptions = model.transcribe(audio_paths)
    print(transcriptions)

    """Create Processor"""

    # timit = timit.map(prepare_dataset, remove_columns=timit.column_names, num_proc=4)
    # timit = timit.filter(lambda x: x < 25 * 16_000, input_columns=["input_length"])

    timit.num_rows

    """Evaluation"""

    references = prep_asr_testing(timit)

    references

    evaluation = model.evaluate(references)

    print("WER AND CER LOCATION")
    print(evaluation)

    test2 = ['content/downloads/Torgo/M04/Session2/wav_headMic/0297.wav','content/downloads/Torgo/M04/Session2/wav_headMic/0298.wav']

    model.transcribe(test2)

    test3 = [{'path': 'content/downloads/Torgo/M04/Session2/wav_headMic/0297.wav',
        'transcription': 'BIT'},
    {'path': 'content/downloads/Torgo/M04/Session2/wav_headMic/0295.wav',
        'transcription': 'THERE IS A TREEHOUSE UP ABOVE'}]


if __name__=="__main__":
    prep_csv("output_og.csv", min_length=1)

    main()
