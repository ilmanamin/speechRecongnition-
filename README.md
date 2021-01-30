# speechRecongnition- Use LINUX only
# idea from Michael Nguyen (LearnedVector)
### using virtualenv (recommend)
1. `virtualenv voiceassistant.venv`
2. `source voiceassistant.venv/bin/activate`

### pip packages
`pip install -r requirements.txt` 

## Speech Recognition
[YouTube Video for Speech Recognition](https://www.youtube.com/watch?v=YereI6Gn3bM&list=PL5rWfvZIL-NpFXM9nFr15RmEEh4F4ePZW&index=2)

### scripts
For more details make sure to visit these files to look at script arguments and description

`speechrecognition/scripts/mimic_create_jsons.py`is used to create the train.json and test.json files with Mimic Recording Studio 

`speechrecognition/scripts/commonvoice_create_jsons.py`is used to convert mp3 into wav and create the train.json and test.json files with the Commonvoice dataset

`spechrecognition/neuralnet/train.py` is used to train the model

`spechrecognition/neuralnet/optimize_graph.py` is used to create a production ready graph that can be used in `engine.py`

`spechrecognition/engine.py` is used to demo the speech recognizer model

`spechrecognition/demo/demo.py` is used to demo the speech recognizer model with a Web GUI


### Steps for pretraining or finetuning speech recognition model

The pretrained model can be found here at this [google drive](https://drive.google.com/drive/folders/14ljfpvisK1tz8fvFYETbdWqR3lOmJ_2Y?usp=sharing)

1. Collect your own data - the pretrain model was trained on common voice. To make this model work for you, you can collect about an hour or so of your own voice using the [Mimic Recording Studio](https://github.com/MycroftAI/mimic-recording-studio). They have prompts that you can read from.
    1. collect data using mimic recording studio, or your own dataset.
    2. be sure to chop up your audio into 5 - 16 seconds chunks max.
    3. create a train and test json in this format...
    ```
        // make each sample is on a seperate line
        {"key": "/path/to/audio/speech.wav, "text": "this is your text"}
        {"key": "/path/to/audio/speech.wav, "text": "another text example"}
    ```
    use `mimic_create_jsons.py` to create train and test json's with the data from Mimic Recording Studio.
        
        python mimic_create_jsons.py --file_folder_directory /dir/to/the/folder/with/the/studio/data --save_json_path /path/where/you/want/them/saved

    (The Mimic Recording Studio files are usually stored in ~/mimic-recording-studio-master/backend/audio_files/[random_string].) 
    
    use `commonvoice_create_jsons.py` to convert from mp3 to wav and to create train and test json's with the data from Commonvoice by Mozilla
        
        python commonvoice_create_jsons.py --file_path /path/to/commonvoice/file/.tsv --save_json_path /path/where/you/want/them/saved 
    
    if you dont want to convert use `--not-convert` 
        
2. Train model
    1. use `train.py` to fine tune. checkout the [train.py](https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant/blob/master/VoiceAssistant/speechrecognition/neuralnet/train.py#L115) argparse for other arguments
    ```
       python train.py --train_file /path/to/train/json --valid_file /path/to/valid/json --load_model_from /path/to/pretrain/speechrecognition.ckpt
    ```
   2. To train from scratch omit the `--load_model_from` argument in train.py
   3. after model training us `optimize_graph.py` to create a frozen optimized pytorch model. The pretrained optimized torch model can be found in the google drive link as `speechrecognition.zip`


3. test
    1. test using the `engine.py` script
