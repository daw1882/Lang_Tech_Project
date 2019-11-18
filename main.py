from emotion_analyzer import *
from record_audio import *
import matplotlib.pyplot as plt
import pygame

display_width = 800
display_height = 600


def text_format(message, textFont, textSize, textColor):
    newFont=pygame.font.SysFont(textFont, textSize)
    newText=newFont.render(message, 0, textColor)

    return newText


def start_menu(window):
    menu = True
    pygame.init()
    font = 'comicsansms'
    pygame.display.set_caption('Emotions')
    while menu:
        window.fill((255, 255, 255))
        text_start = text_format("PRESS ENTER TO START",
                                 font, 50, (0, 0, 0))
        start_msg = text_start.get_rect()
        start_msg.center = (display_width//2, display_height//2)
        window.blit(text_start, start_msg)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                menu = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    menu = False
                    plot_analysis(window)
        pygame.display.update()


def plot_analysis(window):
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    colors = ['#B29DD9', '#FDFD98', '#FFB447', '#FE6B64', '#77DD77',
              '#779ECB', 'pink']
    # Create pie chart to show percentages of emotions
    for i in range(len(labels)):
        labels[i] += ': ' + ('%1.2f%%' % sizes[i])
    patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
    plt.legend(patches, labels, loc="best", facecolor='#f9f9f9', edgecolor='black')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('piechart.png')

    # pygame display

    image = pygame.image.load('piechart.png')
    font = pygame.font.SysFont('comicsansms', 50)
    text = font.render('Top Emotion Detected: ' + top, True, (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.center = (display_width//2, display_height-50)
    window_run = True
    while window_run:
        window.fill((255, 255, 255))
        window.blit(image, (0, 0))
        window.blit(text, text_rect)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                window_run = False
            pygame.display.update()


if __name__ == '__main__':
    running = True
    while running:
        #record_and_save()
        sizes, top = get_emotions('mchocola.wav')
        window_display = pygame.display.set_mode((display_width, display_height))
        start_menu(window_display)
        running = False
