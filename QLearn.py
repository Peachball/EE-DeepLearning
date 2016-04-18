from __future__ import print_function
import time
import pygame
import numpy as np
from graphics import *
from DeepLearning import Layer
from DeepLearning import FFClassifier
from DeepLearning import generateVanillaUpdates
from DeepLearning import generateRpropUpdates
import theano
import theano.tensor as T

width = 800
height = 800

def evalSize(board, update=False):
    hx = 0
    hy = 0
    maxSize = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] > maxSize:
                hx = i
                hy = j
                maxSize = board[i][j]
            if update==True and board[i][j] > 0:
                board[i][j] -= 1
    return [maxSize, (hx, hy)]

def nextIter(board, direction):
    [size, head] = evalSize(board, update=True)
    
    xsize = len(board)
    ysize = len(board[0])
    head = list(head)
    if direction == 0:
        if head[1] + 1 >= ysize:
            head[1] = 0
        else: head[1] += 1
    if direction == 1:
        if head[0] + 1 >= xsize:
            head[0] = 0
        else: head[0] += 1
    if direction == 2:
        if head[1] - 1 < 0:
            head[1] = ysize - 1
        else: head[1] -= 1
    if direction == 3:
        if head[0] - 1 < 0:
            head[0] = xsize - 1
        else: head[0] -= 1
    
    if board[head[0]][head[1]] < 0:
        board[head[0]][head[1]] = size + 1
        return size
    
    if board[head[0]][head[1]] > 0:
        return -size

    board[head[0]][head[1]] = size
    return 0

def addApple(board, random=True):
    if random:
        rng = np.random.randint(len(board) * len(board[0]) - evalSize(board)[0])
    else:
        rng = 1000
    
    spaces = 0
    for i in range(len(board)):
        for j in range(len(board)):
            if spaces >= rng and board[i][j] == 0:
                board[i][j] = -1
                return
            spaces += 1
        if i == len(board) - 1:
            i = 0
    print("Add apple has a problem")

def clear(win):
    win.fill((255, 255, 255))

def createBoard(size):
    board = []
    for i in range(size):
        board.append([])
        for j in range(size):
            board[i].append(0)
    return board

def display(board, win):
    def draw(pos, state):
        sizex =  width / len(board)
        sizey = height / len(board[0])
        color = (0, 0, 0)
        r = Rectangle(Point(sizex * pos[0], sizey * pos[1]), 
                Point(sizex*pos[0] + sizex, sizey * pos[1] + sizey))
        if state == "apple":
            color = (255, 0, 0)
        elif state=='snake':
            color = (0, 0, 0)
        else:
            color = (255, 255, 255)
        pygame.draw.rect(win, color, ((sizex*(pos[0])),(sizex*(pos[1])),sizex, sizey))

    for i in range(len(board)):
        for j in range(len(board)):
            state = 'white'
            if board[i][j] > 0:
                state='snake'
            if board[i][j] < 0:
                state='apple'
            draw((i, j), state)

def convertBoard(board, style='legit'):
    [size, head] = evalSize(board)
    output = [0] * len(board) * len(board[0]) * 2
    if style=='legit':
        for i in range(len(board)):
            for j in range(len(board[0])):
                xpo = i - head[0]
                ypo = j - head[1]
                if xpo < 0:
                    xpo += len(board)
                if ypo < 0:
                    ypo += len(board)
                
                if board[i][j] > 0:
                    output[2 * (xpo * len(board[0]) + ypo)] = board[i][j]
                if board[i][j] < 0:
                    output[2 * (xpo * len(board[0]) + ypo)] = 1
        return output

def predict(prediction, temperature=1):
    total = np.sum(np.power(np.e, prediction/temperature))
    rng = np.random.rand()
    direction = 3
    b = 0
    for i in range(4):
        if rng < b + pow(np.e, prediction[0,i]/temperature)/total:
            direction = i
            break
        b += pow(np.e, prediction[0,i]/temperature)/total
    return direction

def LearnSnake():
    boardSize = 5
    scr = pygame.display.set_mode((width, height))
    pygame.display.flip()
    b = createBoard(boardSize)
    b[0][1] = 1
    addApple(b)
    q1 = FFClassifier(boardSize * boardSize *2, 700, init_size=0.1)
    q2 = Layer(700, 4, layer_type='rlu', init_size=0.1, in_var=q1.out)

    params = q1.params + q2.params
    out = q2.out
    y = T.matrix('output')
    mse = T.mean(T.sqr(out  - y))

    updates = generateVanillaUpdates(params, 0.0001, mse)
    (storage, rupdates) = generateRpropUpdates(params, mse, init_size=0.1)
    learn = theano.function([q1.x, y], mse, updates=updates)
    p = theano.function([q1.x], out)

    nextIter(b, 0)


    print("time to watch some snake")
    #Snake Drawing things
    
    lam = 0.9
    maxMoves = boardSize * boardSize // 2
    mem = []
    cor = []
    maxSize = 0
    maxMoves = 1000
    draw = True
    while True:
        moves = 0
        tooSlow = 0
        while True:
            moves += 1
            bef = np.array(convertBoard(b)).reshape(1, boardSize * boardSize * 2)
            prediction = p(bef)
            move = predict(prediction, temperature=1)

            if draw:
                clear(scr)
                display(b, scr)
                pygame.display.update()
            status = nextIter(b, move)
            after = np.array(convertBoard(b)).reshape(1, boardSize * boardSize * 2)
    
            r = 0
            tooSlow += 1
            if status > 0:
                addApple(b)
                r = 100
                tooSlow = 0
            end = False
            if status < 0 or tooSlow > maxMoves:
                r = 0
                end = True

            #Training part
            correct = prediction
            correct[0,move] = r + lam * np.max(p(after))

            if end:
                correct = np.array([0]*4)

            mem.append(np.squeeze(bef))
            cor.append(np.squeeze(correct))
            print(p(bef))
            if end:
                break
        [size, _] = evalSize(b)
        if size > maxSize:
            maxSize = size
            print(size)
        if len(mem) > 1000:
            error = 10
            for j in range(10):
                error = (learn(np.array(mem), np.array(cor)))
                print(error)
            mem = []
            cor = []
            print(error)
        b = createBoard(boardSize)
        b[0][0] = 1
        addApple(b)


if __name__ == '__main__':
    LearnSnake()
