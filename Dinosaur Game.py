import pygame, random
from colors import *
import numpy as np

pygame.init()

window_width = 600
window_height = 600
game_display = pygame.display.set_mode((window_width, window_height))
clock = pygame.time.Clock()


class Camera:
    def __init__(self):
        self.xoffset = 0
        self.yoffset = 0
        self.speed = 8
        self.oldbest = 0


class Player:
    def __init__(self):
        self.x = 100
        self.size = 30
        self.y = 400 - self.size
        self.speed = 10
        self.jump = False
        self.yVelocity = 0
        self.ground = True

    def draw(self):
        pygame.draw.rect(game_display, brown, [self.x, self.y, self.size, self.size])

    def checkHit(self):
        for x in Obstacles1.obstacles:
            y = Obstacles1.obstacles[x]
            if y.x - Camera1.xoffset <= self.x <= y.x - Camera1.xoffset + y.size:
                if y.y <= self.y + self.size <= y.y + y.size:
                    return True
            if (
                y.x - Camera1.xoffset
                <= self.x + self.size
                <= y.x - Camera1.xoffset + y.size
            ):
                if y.y <= self.y + self.size <= y.y + y.size:
                    return True


class Obstacle:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size


class Obstacles:
    def __init__(self):
        self.obstacles = {}
        self.obstacleAmount = 0

    def makeObstacle(self, x, y):
        self.obstacleAmount += 1
        size = 35
        self.obstacles[str(self.obstacleAmount)] = Obstacle(x, y, size)

    def draw(self):
        for x in self.obstacles:
            y = self.obstacles[x]
            pygame.draw.rect(
                game_display, blue, [y.x - Camera1.xoffset, y.y, y.size, y.size]
            )


def MakeRandGeneration(size):
    Pop = []
    for x in range(size):
        PopGenetics = []
        for y in range(9):
            PopGenetics.append(2 * random.random() - 1)
        Pop.append(PopGenetics)
    return Pop


def draw_ground():
    pygame.draw.rect(
        game_display, limegreen, [0, 400, window_width, window_height - 400]
    )


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def feed_forward(input1, input2, syn0, syn1):
    input1 /= 50.0
    input2 /= 10.0
    l0 = np.array([[input1, input2]])
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    return l2


Player1 = Player()
Obstacles1 = Obstacles()
Size = 20
Population = MakeRandGeneration(Size)


def smallest_distance():
    smallest_dist = 900
    for x in Obstacles1.obstacles:
        y = Obstacles1.obstacles[x]
        if (y.x - Camera1.xoffset) - (Player1.x + Player1.size) < smallest_dist and (
            y.x - Camera1.xoffset
        ) - (Player1.x + Player1.size) > 0:
            smallest_dist = (y.x - Camera1.xoffset) - (Player1.x + Player1.size)
    return smallest_dist


def find_passed_obstacles():
    num = 0
    for x in Obstacles1.obstacles:
        y = Obstacles1.obstacles[x]
        if (y.x - Camera1.xoffset) - (Player1.x + Player1.size) < 0:
            num += 1
    return num


def make_fitness():
    global Camera1, Player1, Obstacles1
    fitness_list = []
    Obstacles1 = Obstacles()
    for i in Population:
        # Start
        spawn_cd = 20
        max_cd = 20
        game_exit = False
        syn0 = np.array([[i[0], i[1], i[2]], [i[3], i[4], i[5]]])
        syn1 = np.array([[i[6]], [i[7]], [i[8]]])
        syn0 = np.array([[0.105, 0.37, -0.48], [-0.46, 0.2, 0.97]])
        syn1 = np.array([[-0.8488], [-0.13], [0.811]])
        Player1 = Player()
        Camera1 = Camera()
        while not game_exit:
            game_display.fill(skyblue)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        print(syn0)
                        print(syn1)
                    if event.key == pygame.K_ESCAPE:
                        Camera1.xoffset = 0
                        Camera1.speed = 8
                        Camera1.oldbest = 0
                        Obstacles1 = Obstacles()
                        value = find_passed_obstacles()
                        if find_passed_obstacles() == 0:
                            value = 1
                        fitness_list.append(Camera1.xoffset * value)
                        game_exit = True
                    if event.key == pygame.K_s:
                        Camera1.speed += 0.25

            draw_ground()
            Player1.draw()
            Obstacles1.draw()
            if not Player1.ground:
                Player1.yVelocity += 1.5
            Player1.y += Player1.yVelocity
            if random.randint(1, 60) == 1 and spawn_cd == max_cd:
                spawn_cd = 0
                Obstacles1.makeObstacle(Camera1.xoffset + 800, 365)
            if Player1.y >= 400 - Player1.size:
                Player1.ground = True
                Player1.y = 400 - Player1.size
            Camera1.xoffset += Camera1.speed
            if (Camera1.xoffset / 200) - (Camera1.xoffset % 200) > Camera1.oldbest:
                Camera1.speed += 0.25
                Camera1.oldbest = Camera1.xoffset / 200
            if spawn_cd != max_cd:
                spawn_cd += 1
            if Player1.checkHit():
                Camera1.xoffset = 0
                Camera1.speed = 8
                Camera1.oldbest = 0
                Obstacles1 = Obstacles()
                value = find_passed_obstacles()
                if find_passed_obstacles() == 0:
                    value = 1
                fitness_list.append(Camera1.xoffset * value)
                game_exit = True
                print("dead")
            if smallest_distance() != 900:
                if (
                    feed_forward(smallest_distance(), Camera1.speed, syn0, syn1)[0][0]
                    > 0.5
                    and Player1.ground == True
                ):
                    Player1.jump = True
                    Player1.ground = False
                    Player1.yVelocity = -18
            pygame.display.update()
            clock.tick(45)
    return fitness_list


for x in range(5):
    PopFitness = make_fitness()
    print(PopFitness)
pygame.quit()
quit()
# End
