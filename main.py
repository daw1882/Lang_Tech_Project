from emotion_analyzer import *
from record_audio import *
import matplotlib.pyplot as plt
import pygame

pygame.init()
COLOR_INACTIVE = pygame.Color('black')
COLOR_ACTIVE = pygame.Color('black')
FONT = pygame.font.SysFont('comicsansms', 32)
FILENAME = ''


class InputBox:

    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = COLOR_INACTIVE
        self.text = text
        self.txt_surface = FONT.render(text, True, COLOR_ACTIVE)
        self.active = False

    def handle_event(self, event):
        global FILENAME
        if event.type == pygame.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = True
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    print(self.text)
                    FILENAME = self.text
                    self.text = ''
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = FONT.render(self.text, True, COLOR_ACTIVE)

    def update(self):
        # Resize the box if the text is too long.
        width = max(self.rect.w, self.txt_surface.get_width()+10)
        self.rect.w = width

    def draw(self, screen):
        # Blit the text.
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y-5))
        # Blit the rect.
        pygame.draw.rect(screen, self.color, self.rect, 2)


display_width = 800
display_height = 600


def text_format(message, textFont, textSize, textColor):
    newFont = pygame.font.SysFont(textFont, textSize)
    newText = newFont.render(message, True, textColor)
    return newText


def start_menu(window):
    global FILENAME
    menu = True
    font = 'comicsansms'
    pygame.display.set_caption('Emotions')
    opt1_text = text_format("Press '1' to record your own file",
                            font, 40, (0, 0, 0))
    opt2_text = text_format("or press '2' to input the name of a file",
                            font, 40, (0, 0, 0))
    opt1_rect = opt1_text.get_rect()
    opt1_rect.center = (display_width // 2, display_height // 2 - 50)
    opt2_rect = opt1_text.get_rect()
    opt2_rect.center = (display_width // 2 - 50, display_height // 2 + 25)
    while menu:
        window.fill((255, 255, 255))
        window.blit(opt1_text, opt1_rect)
        window.blit(opt2_text, opt2_rect)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                menu = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    # record_and_save()
                    FILENAME = 'output.wav'
                    menu = False
                if event.key == pygame.K_2:
                    get_input_file(window)
                    menu = False

        pygame.display.update()


def get_input_file(window):
    global FILENAME
    input_box1 = InputBox(25, display_height//2, 750, 40)
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            input_box1.handle_event(event)
            input_box1.update()

        window.fill((255, 255, 255))
        input_box1.draw(window)
        if FILENAME != '':
            done = True

        pygame.display.flip()


def plot_analysis(window):
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    colors = ['#B29DD9', '#FDFD98', '#FFB447', '#FE6B64', '#77DD77',
              '#779ECB', 'pink']
    font = 'comicsansms'
    # Create pie chart to show percentages of emotions
    for i in range(len(labels)):
        labels[i] += ': ' + ('%1.2f%%' % sizes[i])
    patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
    plt.legend(patches, labels, loc="upper left", facecolor='#f9f9f9', edgecolor='black')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('piechart.png')

    image = pygame.image.load('piechart.png')
    text = text_format('Top Emotion Detected: ' + top, font, 40, (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.center = (display_width//2, display_height-50)

    question1 = text_format('Press ESC to exit', font, 20, (0, 0, 0))
    question_rect1 = question1.get_rect()
    question_rect1.center = ((4/5)*display_width, display_height//2 - 100)

    question2 = text_format('or ENTER to rerun', font, 20, (0, 0, 0))
    question_rect2 = question1.get_rect()
    question_rect2.center = ((4/5) * display_width, display_height // 2 - 75)

    window_run = True
    while window_run:
        window.fill((255, 255, 255))
        window.blit(image, (0, 0))
        window.blit(text, text_rect)
        window.blit(question1, question_rect1)
        window.blit(question2, question_rect2)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                window_run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_RETURN:
                    return True
        pygame.display.update()


if __name__ == '__main__':
    running = True
    while running:
        window_display = pygame.display.set_mode((display_width, display_height))
        start_menu(window_display)
        sizes, top = get_emotions(FILENAME)
        FILENAME = ''
        running = plot_analysis(window_display)
