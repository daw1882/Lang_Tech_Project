import pyaudio
import wave
import keyboard
import time
import pygame


def text_format(message, textFont, textSize, textColor):
    newFont = pygame.font.SysFont(textFont, textSize)
    newText = newFont.render(message, True, textColor)
    return newText


def record_and_save(window):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 12200  # Record at 44100 samples per second
    seconds = 3
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
    running = True
    time.sleep(0.5)

    print("Recording. Press SPACE to stop.")

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    while running:
        data = stream.read(chunk)
        frames.append(data)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                menu = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = False

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording.')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

