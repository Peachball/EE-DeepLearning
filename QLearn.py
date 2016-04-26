from __future__ import print_function
import random
import time
import pygame
import numpy as np
from graphics import *
from DeepLearning import Layer
from DeepLearning import FFClassifier
from DeepLearning import generateVanillaUpdates
from DeepLearning import generateRpropUpdates
from DeepLearning import generateMomentumUpdates
from DeepLearning import generateRmsProp
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
        return size+1
    
    if board[head[0]][head[1]] > 0:
        board[head[0]][head[1]] = size
        return -(size)

    board[head[0]][head[1]] = size
    return 0

def addApple(board, random=True):
    if random:
        rng = np.random.randint(2 * len(board) * len(board[0]) - evalSize(board)[0])
    else:
        rng = 1000
    
    spaces = 0
    i = -1
    while spaces <= rng:
        i += 1
        for j in range(len(board)):
            if board[i][j] > 0:
                continue
            if spaces == rng:
                board[i][j] = -1
                return 0
            spaces += 1
        if i == len(board) - 1:
            i = 0
    return 1

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
    [_, (headx, heady)] = evalSize(board)
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
        elif state=='head':
            color = (0, 255, 0)
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
            if i == headx and j == heady:
                state='head'
            draw((i, j), state)

def convertBoard(board, style='legit'):
    [size, head] = evalSize(board)
    if style=='legit':
        output = [0] * len(board) * len(board[0]) * 2
        for i in range(len(board)):
            for j in range(len(board[0])):
                xpo = i - head[0]
                ypo = j - head[1]
                if xpo < 0:
                    xpo += len(board)
                if ypo < 0:
                    ypo += len(board[0])
                
                if board[i][j] > 0:
                    output[2 * (xpo * len(board[0]) + ypo)] = board[i][j]
                if board[i][j] < 0:
                    output[2 * (xpo * len(board[0]) + ypo) + 1] = 1
        return output
    
    if style=='shady':
        output = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] > 0:
                    output.append(board[i][j])
                    output.append(0)
                    [_, head] = evalSize(board)
                    if head[0] == i and head[1] == j:
                        output.append(1)
                    else:
                        output.append(0)
                elif board[i][j] < 0:
                    output.append(0)
                    output.append(1)
                    output.append(0)
                else:
                    output.append(0)
                    output.append(0)
                    output.append(0)
        return output


def predict(prediction, temperature=1, noDir=-1):
    total = np.sum(np.power(np.e, prediction/temperature))
    if noDir >= 0:
        noDir = (noDir + 2) % 4
        total -= np.power(np.e, prediction[0, noDir]/temperature)
    rng = np.random.rand()
    direction = 3
    b = 0
    for i in range(4):
        if i == noDir:
            continue
        if rng < b + pow(np.e, prediction[0,i]/temperature)/total:
            direction = i
            break
        b += pow(np.e, prediction[0,i]/temperature)/total
    return direction

def LearnSnake():
    boardSize = 10
    scr = pygame.display.set_mode((width, height))
    b = createBoard(boardSize)
    b[0][1] = 1
    addApple(b)
    q1 = FFClassifier(boardSize * boardSize * 2, 700, 300, init_size=0.01)
    q2 = Layer(300, 4, layer_type='linear', init_size=0.01, in_var=q1.out)

    x = T.matrix('input')
    q3 = Layer(boardSize * boardSize * 2, 4, init_size=0.01, layer_type='linear', in_var=x)

    params = q1.params + q2.params
    out = q2.out
    y = T.matrix('output')
    mse = T.mean(T.sqr(out  - y))

    updates = generateVanillaUpdates(params, 0.01, mse)
    (storage, rupdates) = generateRpropUpdates(params, mse, init_size=0.01)
    (storage, mupdates) = generateMomentumUpdates(params, 0.5, 0.01, mse)
    ([_, _, alpha], rms) = generateRmsProp(params, 0.01, 0.9, mse)
    learn = theano.function([q1.x, y], mse, updates=rms)
    rlearn = theano.function([q1.x, y], mse, updates=mupdates)
    p = theano.function([q1.x], out)

    print("time to watch some snake")
    #Snake Drawing things
    
    lam = 0.9
    maxMoves = boardSize * boardSize // 2
    mem = []
    rew = []
    act = []
    nstate = []
    maxSize = 0
    maxMem = 10000
    draw = False
    style = 'legit'
    maxMoves = -1
    if style == 'shady':
        inputsize = 3 * boardSize ** 2
    elif style=='legit':
        inputsize = 2 * boardSize**2
    temp = 1
    while True:
        moves = 0
        tooSlow = 0
        prevMove = -1
        while True:
            moves += 1
            bef = np.array(convertBoard(b, style=style)).reshape(1, inputsize)
            prediction = p(bef)
            move = predict(prediction, temperature=temp, noDir=-1)
            prevMove = move
            if draw:
                clear(scr)
                display(b, scr)
                pygame.display.update()
                time.sleep(0.1)
            status = nextIter(b, move)
            after = np.array(convertBoard(b, style=style)).reshape(1, inputsize)
    
            r = 0
            tooSlow += 1
            if status > 0:
                if addApple(b) == 1:
                    print("AI Won!")
                    break
                r = 1
                tooSlow = 0
            end = False
            if status < 0 or tooSlow > maxMoves > 0:
                if status < 0:
                    r = -1
                end = True

            mem.append(np.squeeze(bef))
            nstate.append(np.squeeze(after))
            act.append(move)
            rew.append(r)
#            print(p(bef).max() - p(bef).min())
#            print(p(bef))
            if end:
                break
        [size, _] = evalSize(b)
        if size > maxSize:
            maxSize = size
            print("Max Size: ", size)
            print("Temperature: ", temp)

            #Play a sample game of snake
            status = 0
            board = createBoard(boardSize)
            board[0][0] = 1
            addApple(board)
            prevMove = -1
            while status >= 0:
                clear(scr)
                display(board, scr)
                pygame.display.flip()
                time.sleep(0.1)
                c = np.array(convertBoard(board, style=style)).reshape(1, inputsize)
                move = predict(p(c), temperature=temp, noDir=prevMove)
                prevMove = move
                status = nextIter(board, move)
                if status > 0:
                    if addApple(board) == 1:
                        break

        if len(mem) > 1000:
            def pickSamples(size=100):
                samples = random.sample(range(len(mem)), size)
                
                return ([mem[i] for i in samples], 
                        [rew[i] for i in samples],
                        [act[i] for i in samples],
                        [nstate[i] for i in samples])
            def getTarget(m, r, a, s):
                cor = p(np.array(m))
                cor[range(len(cor)),a] = np.greater_equal(r, 0) * (lam * np.max(p(np.array(s)),
                    axis=1)) + r
                '''
                print(cor[range(len(cor)), a])
                print("Prediction:", p(np.array(s)))
                print("Rewards: ", r)
                print("Actions: ", a)
                print("Correct: ", cor)
                raw_input('hm')
                '''
                return cor
            
            def genData(size=100):
                data = pickSamples(size)
                return data[0], getTarget(data[0], data[1], data[2], data[3])
            error = 10
            for j in range(100):
                a, b = genData(size=32)
#                a, b = mem, getTarget(mem, rew, act, nstate)
                a = np.array(a)
                b = np.array(b)
                error = (learn(a, b))
#                print(error)
            '''
            mem = []
            rew = []
            act = []
            nstate = []
            '''
            if temp > 0.1:
                temp *= 0.999
        if len(mem) > maxMem:
            del mem[:-maxMem]
            del rew[:-maxMem]
            del act[:-maxMem]
            del nstate[:-maxMem]
        b = createBoard(boardSize)
        b[0][0] = 1
        addApple(b)


if __name__ == '__main__':
    LearnSnake()
