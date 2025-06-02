
'''
---------------------  2D - геометричні перетворення ---------------------------
Завдання:
Програма повинна будувати 2D графічний об’єкт та реалізовувати його перетворення:
1. Переміщення, як тиражування зображення та в режимі анімації;
2. Обертання, як тиражування зображення та в режимі анімації;
3. Перетворення реалізувати в скалярній та матричній формах.


Варіант 4:
Реалізувати операції: обертання –
переміщення – масштабування.
3. операцію реалізувати циклічно,
траєкторію зміни положення цієї
операції відобразити.
Обрати самостійно: бібліотеку,
розмір графічного вікна, розмір
фігури, параметри реалізації
операцій, кольорову гамму усіх
графічних об’єктів. Всі операції
перетворень мають здійснюватись
у межах графічного вікна.


'''


from graphics import *
import time
import numpy as np
import math as mt

#---------------- Формування та відображення статичного ромба ------------------------

xw = 600; yw = 600; st = 100  # Розміри графічного вікна та параметри перетворень

# Розміри ромба
x1 = xw//2; y1 = yw//2 - st
x2 = xw//2 + st; y2 = yw//2
x3 = xw//2; y3 = yw//2 + st
x4 = xw//2 - st; y4 = yw//2

def draw_initial_shape(win):
    obj = Polygon(Point(x1, y1), Point(x2, y2), Point(x3, y3), Point(x4, y4))
    obj.setFill('blue')
    obj.draw(win)
    return obj

#------------------------- Циклічне обертання ромба ------------------------

def rotate_shape(win, obj):
    win.setBackground('white')
    coords = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1], [x4, y4, 1]])
    center_x, center_y = xw // 2, yw // 2

    def rotate(coords, angle, center):
        angle = mt.radians(angle)
        cx, cy = center
        transformation_matrix = np.array([
            [mt.cos(angle), -mt.sin(angle), 0],
            [mt.sin(angle), mt.cos(angle), 0],
            [0, 0, 1]
        ])
        translated_coords = coords - np.array([cx, cy, 0])
        rotated_coords = translated_coords.dot(transformation_matrix.T)
        final_coords = rotated_coords + np.array([cx, cy, 0])
        return final_coords

    for _ in range(72):
        if win.checkMouse():
            obj.undraw()
            return
        time.sleep(0.1)
        obj.undraw()
        coords = rotate(coords, 5, (center_x, center_y))
        obj = Polygon(Point(coords[0, 0], coords[0, 1]),
                      Point(coords[1, 0], coords[1, 1]),
                      Point(coords[2, 0], coords[2, 1]),
                      Point(coords[3, 0], coords[3, 1]))
        obj.setFill('green')
        obj.draw(win)
    obj.undraw()

#------------------------- Переміщення ромба ------------------------

def move_shape(win):
    win.setBackground('white')
    obj = draw_initial_shape(win)
    dx, dy = 5, 5

    for _ in range(50):
        if win.checkMouse():
            obj.undraw()
            return
        time.sleep(0.1)
        obj.move(dx, dy)
    obj.undraw()

#------------------------- Масштабування ромба ------------------------

def scale_shape(win):
    win.setBackground('white')
    obj = draw_initial_shape(win)
    coords = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1], [x4, y4, 1]])
    center_x, center_y = xw // 2, yw // 2
    scale_factor = 1.05
    scale_limit = 1.5
    scale_step = 0.05
    scale_cycles = 3

    for _ in range(scale_cycles):
        increasing = True
        for step in range(12):
            if win.checkMouse():
                obj.undraw()
                return
            time.sleep(0.2)
            obj.undraw()
            factor = scale_factor if increasing else 1 / scale_factor
            scaling_matrix = np.array([
                [factor, 0, center_x * (1 - factor)],
                [0, factor, center_y * (1 - factor)],
                [0, 0, 1]
            ])
            coords = coords.dot(scaling_matrix.T)
            obj = Polygon(Point(coords[0, 0], coords[0, 1]),
                          Point(coords[1, 0], coords[1, 1]),
                          Point(coords[2, 0], coords[2, 1]),
                          Point(coords[3, 0], coords[3, 1]))
            obj.setFill('red')
            obj.draw(win)

            if step == 5:
                increasing = False
        obj.undraw()

win = GraphWin("2-D трансформації", xw, yw)
obj = draw_initial_shape(win)
win.getMouse()
obj.undraw()
rotate_shape(win, obj)
win.getMouse()
move_shape(win)
win.getMouse()
scale_shape(win)
win.getMouse()
win.close()
