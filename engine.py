#stt1
import pyaudio
import threading
import time
import argparse
import wave
import torchaudio
import torch
import sys
import os
import datetime
import numpy as np
from neuralnet.dataset import get_featurizer
from decoder import DecodeGreedy, CTCBeamDecoder
from threading import Event

from fastpunct import FastPunct


#bot
import requests
#import speech_recognition as sr
import subprocess
from gtts import gTTS
from playsound import playsound
from mpyg321.mpyg321 import MPyg321Player



class Listener:

    def __init__(self, sample_rate=8000, record_seconds=5):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        output=True,
                        frames_per_buffer=self.chunk)

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk , exception_on_overflow=True)
            queue.append(data)
            time.sleep(0.01)

    def close(self):
        self._stream.close()
        self._pa.terminate()

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\Speech Recognition engine is now listening... \n")


class SpeechRecognitionEngine:

    def __init__(self, model_file, ken_lm_file, context_length=10):
        self.listener = Listener(sample_rate=8000)
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')  #run on cpu
        self.featurizer = get_featurizer(8000)
        self.audio_q = list()
        self.hidden = (torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024))
        self.beam_results = ""
        self.out_args = None
        self.beam_search = CTCBeamDecoder(beam_size=100, kenlm_path=ken_lm_file)
        self.context_length = context_length * 50 # multiply by 50 because each 50 from output frame is 1 second
        self.start = False

    def saved(self, waveforms,ilman, fname="audio_temp"):
        wf = wave.open(fname +str(ilman) , "wb")
        # set the channels
        wf.setnchannels(1)
        # set the sample format
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        # set the sample rate
        wf.setframerate(8000)
        # write the frames as bytes
        wf.writeframes(b"".join(waveforms))
        # close the file
        wf.close()
        return fname

    def save(self, waveforms, fname="audio_temp"):
        wf = wave.open(fname , "wb")
        # set the channels
        wf.setnchannels(1)
        # set the sample format
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        # set the sample rate
        wf.setframerate(8000)
        # write the frames as bytes
        wf.writeframes(b"".join(waveforms))
        # close the file
        wf.close()
        return fname

    def predict(self, audio,ilman):
        with torch.no_grad():
            xname = self.saved(audio,ilman)
            fname = self.save(audio)
            waveform, _ = torchaudio.load(fname)  # don't normalize on train
            log_mel = self.featurizer(waveform).unsqueeze(1)
            out, self.hidden = self.model(log_mel, self.hidden)
            out = torch.nn.functional.softmax(out, dim=2)
            out = out.transpose(0, 1)
            self.out_args = out if self.out_args is None else torch.cat((self.out_args, out), dim=1)
            results = self.beam_search(self.out_args)
            current_context_length = self.out_args.shape[1] / 50  # in seconds
            if self.out_args.shape[1] > self.context_length:
                self.out_args = None
            return results, current_context_length

    def inference_loop(self, action):
        endTime = datetime.datetime.now() + datetime.timedelta(seconds=8)
        ilman =0
        while datetime.datetime.now() <= endTime:

            if len(self.audio_q) < 5.5:
                continue
            else:
                ilman=ilman+1
                pred_q = self.audio_q.copy()
                self.audio_q.clear()
                action(self.predict(pred_q,ilman))


                print(ilman)
            time.sleep(0.05)
        print("end")



    def run(self, action):            
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop,
                                    args=(action,), daemon=True)
        thread.start()



class DemoAction:

    def __init__(self):
        self.asr_results = ""
        self.current_beam = ""

    def __call__(self, x):
        results, current_context_length = x
        self.current_beam = results
        trascript = " ".join(self.asr_results.split() + results.split())
        print(trascript)
        self.save_transcript(trascript)
        if current_context_length > 10:
            self.asr_results = trascript

    def save_transcript(self, transcript):
        with open("transcript.txt", 'w+') as f:
            f.write(transcript)

#######################################################STARTS HERE###############################################################################################################
bot_message = ""
message = ""

while (1) :
    #stt1
    
    asr_engine = SpeechRecognitionEngine("","")
    fastpunct = FastPunct()
    action = DemoAction()
    asr_engine.run(action)
    threading.Event().wait(10)
    with open('transcript.txt', 'r') as f:
        message = f.read()
        print("Message before: "+message)
    seperate = ', '
    messagePunct = seperate.join(fastpunct.punct([message]))
    print("Message after: "+messagePunct)
    os.remove("transcript.txt")

    
###################################################################END HERE######################################################################################################








