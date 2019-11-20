import pyaudio
import wave
import time
import pygame


def text_format(message, textFont, textSize, textColor):
    newFont = pygame.font.SysFont(textFont, textSize)
    newText = newFont.render(message, True, textColor)
    return newText


def record_and_save(window):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 2
    fs = 12200
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    text = text_format('Press SPACE to start recording', 'comicsansms', 40, (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.center = (400, 300)
    waiting = True
    while waiting:
        window.fill((255, 255, 255))
        window.blit(text, text_rect)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
        pygame.display.update()
    running = True

    time.sleep(0.5)
    print("Recording. Press SPACE to stop.")

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames
    text = text_format('Recording...Press SPACE to stop', 'comicsansms', 40, (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.center = (400, 300)
    while running:
        window.fill((255, 255, 255))
        window.blit(text, text_rect)
        data = stream.read(chunk)
        frames.append(data)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = False
        pygame.display.update()

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print('Finished recording.')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
