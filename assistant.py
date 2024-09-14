from dotenv import load_dotenv
from groq import Groq
import os
from PIL import ImageGrab, Image
import google.generativeai as genai
import pyperclip
import cv2
import pyttsx3
import speech_recognition as sr
from datetime import datetime  

AI_NAME = 'Jarvis'
USER_NAME = "Joel Thomas"

load_dotenv()

groq_client = Groq(api_key=os.getenv("groq_api"))
genai.configure(api_key=os.getenv("google_api"))
web_cam = cv2.VideoCapture(0)

def get_timestamp():
    return datetime.now().strftime("%I:%M %p on %A, %d %B %Y")

sys_msg = (
    f'You are multi-modal AI voice assistant. of {USER_NAME} created by him and your name is {AI_NAME}. Your user may or may not have attached a photo for context '
    f'(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    f'text prompt that will be attached to their transcribed voice prompt. And never tell it as an image just say that from what I see. It will be the real-time context. Generate the most useful and '
    f'factual response possible, carefully considering all previous generated text in your response before '
    f'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    f'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    f'your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
    'temperature': 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048
}

safety_settings = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                             generation_config=generation_config,
                             safety_settings=safety_settings)

def groq_prompt(prompt, img_context):
    timestamped_prompt = f"{get_timestamp()} - {prompt}"
    if img_context:
        prompt = f"USER PROMPT: {timestamped_prompt}\n\nIMAGE CONTEXT: {img_context}"
    else:
        prompt = f"USER PROMPT: {timestamped_prompt}"
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append({'role': response.role, 'content': response.content})
    return response.content

engine = pyttsx3.init('sapi5')

def speak(text):
    print(f"{AI_NAME}: {text}")
    engine.say(text)
    engine.runAndWait()

def function_call(prompt):
    timestamped_prompt = f"{get_timestamp()} - {prompt}"
    sys_msg = (
        'You are an AI function-calling model. You will determine whether extracting the user\'s clipboard content, '
        'taking a screenshot, capturing the webcam, or calling no functions is best for a voice assistant to respond '
        'to the user\'s prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from the list: ["extract clipboard", "take screenshot", "capture webcam", "None"].\n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the function call name exactly as listed. don\'t always check clipboard without asking for it'
    )
    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': timestamped_prompt}]
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)

def web_cam_capture():
    if not web_cam.isOpened():
        print('Error: Camera did not open successfully')
        exit()
    path = 'webcam.jpg'
    _, frame = web_cam.read()
    cv2.imwrite(path, frame)

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('No clipboard content')
        return None

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead, take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text

def speech_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print("User: " + text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand the audio.")
            return None
        except sr.RequestError:
            print("Sorry, my speech service is down.")
            return None

vision_context = None

while True:
    prompt = speech_text()

    if prompt is None:
        continue

    call = function_call(prompt)

    if 'take screenshot' in call:
        print("Checking the screen...")
        take_screenshot()
        vision_context = vision_prompt(prompt=prompt, photo_path='screenshot.jpg')
    elif 'capture webcam' in call:
        print('Accessing webcam...')
        web_cam_capture()
        vision_context = vision_prompt(prompt=prompt, photo_path='webcam.jpg')
    elif 'extract clipboard' in call:
        print('Checking clipboard...')
        paste = get_clipboard_text()
        if paste:
            prompt = f"{prompt}\n\nCLIPBOARD CONTENT: {paste}"
        vision_context = None

    response = groq_prompt(prompt=prompt, img_context=vision_context)
    speak(response)
