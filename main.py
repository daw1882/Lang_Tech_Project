from emotion_analyzer import *
from record_audio import *
import matplotlib.pyplot as plt
import pygame

if __name__ == '__main__':
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    colors = ['#B29DD9', '#FDFD98', '#FFB447', '#FE6B64', '#77DD77',
              '#779ECB', 'pink']
    running = True
    while running:
        #record_and_save()
        sizes, top = get_emotions('output.wav')

        # Create pie chart to show percentages of emotions
        for i in range(len(labels)):
            labels[i] += ': ' + ('%1.2f%%' % sizes[i])
        patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
        plt.legend(patches, labels, loc="best", facecolor='#f9f9f9', edgecolor='black')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('piechart.png')

        # pygame display
        pygame.init()
        display_width = 800
        display_height = 600
        window_display = pygame.display.set_mode((display_width, display_height))
        pygame.display.set_caption('Emotions')
        image = pygame.image.load('piechart.png')
        font = pygame.font.SysFont('comicsansms', 50)
        text = font.render('Top Emotion Detected: ' + top, True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (display_width//2, display_height-50)
        window_run = True
        while window_run:
            window_display.fill((255, 255, 255))
            window_display.blit(image, (0, 0))
            window_display.blit(text, text_rect)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    window_run = False
                pygame.display.update()

        running = False
