import speech_recognition as sr
import pvporcupine
import struct
import pyaudio
import multiprocessing
from playsound import playsound
import threading
import nltk
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import pyttsx3
warnings.filterwarnings('ignore')

print('Downloading NTLK Packages')
nltk.download('popular', quiet=True) # for downloading popular packages
nltk.download('punkt') 
nltk.download('wordnet') 
print("Download Complete")

engine = pyttsx3.init() # Init TTS
engine.setProperty('rate', 125)
voices = engine.getProperty('voices') 
engine.setProperty('voice', voices[0].id) 

#Reading in the corpus
with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


r = sr.Recognizer()
mic = sr.Microphone()
with mic as source:
    r.adjust_for_ambient_noise(source, duration=3)
print("Ready")

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating Chatbot Response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


def playCustomSound(sound = 'sounds/wake.mp3'):
    threading.Thread(target=playsound, args=(sound,), daemon=True).start()


def getVoiceInput():
    with mic as source:
        audio = r.listen(source)
    playCustomSound('sounds/correct.mp3')
    google_out = r.recognize_google(audio, language='en-US', show_all=True)
    if (isinstance(google_out, dict)):
        user_response = google_out.get('alternative')[0].get('transcript')
        user_response=user_response.lower()
        print(user_response)
        if(user_response=='thanks' or user_response=='thank you' ):
            return "You are welcome"
        elif(user_response=='i love you'):
            return "I love you, too."
        elif(user_response=='bye'):
            exit()
        else:
            if(greeting(user_response)!=None):
                return greeting(user_response)
            else:
                res = response(user_response)
                sent_tokens.remove(user_response)
                return res
    else:
        return "I didn't quite get that"

porcupine = None
pa = None
audio_stream = None


try:
    porcupine = pvporcupine.create(keywords=["jarvis"])

    pa = pyaudio.PyAudio()

    audio_stream = pa.open(
                    rate=porcupine.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=porcupine.frame_length)

    while True:
        pcm = audio_stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            playCustomSound()
            print("Hotword Detected")
            output = getVoiceInput()
            print(output)
            engine.say(output)
            engine.runAndWait()
finally:
    if porcupine is not None:
        porcupine.delete()

    if audio_stream is not None:
        audio_stream.close()

    if pa is not None:
            pa.terminate()
