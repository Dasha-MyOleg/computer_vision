
'''
---------------  3D - геометричні перетворення та растеризація ---------------------------
Завдання:
Синтезувати 3D об'єкту та здійснити його геометричне перетворення, провести растеризацію:
1. Синтез куба геометричним методом - матриця;
2. Переміщення куба до центру екрану;
3. Обертання навколо пласких координатних вісій;
4. Ортогональна проекція на плаский екран - елюзія 3D об'єкту;
5. Растерізація векторної геометрії 3D куба

'''

from graphics import *
import numpy as np
import math as mt
import matplotlib.colors as colors

#---------------------------------- координати паралелепіпеда ------------------------------------
xw=600; yw=600; st=300
# розташування координат у строках: дальній чотирикутник - A B I M,  ближній чотирикутник D C F E
Prlpd = np.array([ [0, 0, 0, 1],
                  [st, 0, 0, 1],
                  [st, st, 0, 1],
                  [0, st, 0, 1],
                  [0, 0, st, 1],
                  [st, 0, st, 1],
                  [st, st, st, 1],
                  [0, st, st, 1]]  )    # по строках
print('enter matrix')
print(Prlpd)

#--------------------------------- функция проекції на xy, z=0 -------------------------------------
def ProjectXY (Figure):
   f = np.array([ [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1] ])    # по строках
   ft=f.T
   Prxy = Figure.dot(ft)
   print('проекция на ху')
   print(Prxy)
   return Prxy

#-------------------------------------------- зміщення ----------------------------------------------
def ShiftXYZ (Figure, l, m, n):
   f = np.array([ [1, 0, 0, l], [0, 1, 0, m], [0, 0, 1, n], [1, 0, 0, 1] ])    # по строках
   ft=f.T
   Prxy = Figure.dot(ft)
   print('зміщення')
   print(Prxy)
   return Prxy

#-------------------------------------------- обертання коло х----------------------------------------
def insertX (Figure, TetaG):
    TetaR=(3/14*TetaG)/180
    f = np.array([ [1, 0, 0, 0], [0, mt.cos(TetaR), mt.sin(TetaR), 0], [0, -mt.sin(TetaR),  mt.cos(TetaR), 0], [0, 0, 0, 1] ])
    ft=f.T
    Prxy = Figure.dot(ft)
    print('обертання коло х')
    print(Prxy)
    return Prxy

#-------------------------------------------- аксонометрія ----------------------------------------------
def dimetri (Figure, TetaG1, TetaG2):
    TetaR1=(3/14*TetaG1)/180; TetaR2=(3/14*TetaG2)/180;
    f1 = np.array([[mt.cos(TetaR1), 0 , -mt.sin(TetaR1), 0], [0, 1, 0, 0],  [mt.sin(TetaR1), 0, mt.cos(TetaR1), 1], [0, 0, 0, 0],])
    ft1 = f1.T
    Prxy1 = Figure.dot(ft1)
    f2 = np.array([ [1, 0, 0, 0], [0, mt.cos(TetaR2), mt.sin(TetaR2), 0], [0, -mt.sin(TetaR2),  mt.cos(TetaR2), 0], [0, 0, 0, 1] ])
    ft2=f2.T
    Prxy2 = Prxy1.dot(ft2)
    print('dimetri')
    print(Prxy2)
    return Prxy2

#-------------------- функція побудови векторного паралелепіпеда -МОНОХРОМ-------------
def PrlpdWiz(Prxy):
    Ax = Prxy[0, 0];  Ay = Prxy[0, 1]
    Bx = Prxy[1, 0];  By = Prxy[1, 1]
    Ix = Prxy[2, 0];  Iy = Prxy[2, 1]
    Mx = Prxy[3, 0];  My = Prxy[3, 1]

    Dx = Prxy[4, 0];  Dy = Prxy[4, 1]
    Cx = Prxy[5, 0];  Cy = Prxy[5, 1]
    Fx = Prxy[6, 0];  Fy = Prxy[6, 1]
    Ex = Prxy[7, 0];  Ey = Prxy[7, 1]

    #----------- Тильна грань ----------------------------------------------------
    obj = Polygon(Point(Ax, Ay), Point(Bx, By), Point(Ix, Iy), Point(Mx, My))
    obj.draw(win)
    # ----------- Фронтальна грань ------------------------------------------------
    obj = Polygon(Point(Dx, Dy), Point(Cx, Cy), Point(Fx, Fy), Point(Ex, Ey))
    obj.draw(win)
    # ----------- Верхня грань ----------------------------------------------------
    obj = Polygon(Point(Ax, Ay), Point(Bx, By), Point(Cx, Cy), Point(Dx, Dy))
    obj.draw(win)
    # ----------- Нижня грань ----------------------------------------------------
    obj = Polygon(Point(Mx, My), Point(Ix, Iy), Point(Fx, Fy), Point(Ex, Ey))
    obj.draw(win)
    # ----------- Ліва грань ----------------------------------------------------
    obj = Polygon(Point(Ax, Ay), Point(Mx, My), Point(Ex, Ey), Point(Dx, Dy))
    obj.draw(win)
    # ----------- Права грань ----------------------------------------------------
    obj = Polygon(Point(Bx, By), Point(Ix, Iy), Point(Fx, Fy), Point(Cx, Cy))
    obj.draw(win)
    return PrlpdWiz

#-------------------- функція побудови векторного паралелепіпеда -КОЛОР------------
def PrlpdWizRRR(Prxy):
    Ax = Prxy[0, 0];  Ay = Prxy[0, 1]
    Bx = Prxy[1, 0];  By = Prxy[1, 1]
    Ix = Prxy[2, 0];  Iy = Prxy[2, 1]
    Mx = Prxy[3, 0];  My = Prxy[3, 1]

    Dx = Prxy[4, 0];  Dy = Prxy[4, 1]
    Cx = Prxy[5, 0];  Cy = Prxy[5, 1]
    Fx = Prxy[6, 0];  Fy = Prxy[6, 1]
    Ex = Prxy[7, 0];  Ey = Prxy[7, 1]

    #----------- Тильна грань ----------------------------------------------------
    obj = Polygon(Point(Ax, Ay), Point(Bx, By), Point(Ix, Iy), Point(Mx, My))
    obj.setFill('blue')
    obj.draw(win)
    # ----------- Фронтальна грань ------------------------------------------------
    obj = Polygon(Point(Dx, Dy), Point(Cx, Cy), Point(Fx, Fy), Point(Ex, Ey))
    #obj.setFill('green')
    obj.draw(win)
    # ----------- Верхня грань ----------------------------------------------------
    obj = Polygon(Point(Ax, Ay), Point(Bx, By), Point(Cx, Cy), Point(Dx, Dy))
    obj.setFill('violet')
    obj.draw(win)
    # ----------- Нижня грань ----------------------------------------------------
    obj = Polygon(Point(Mx, My), Point(Ix, Iy), Point(Fx, Fy), Point(Ex, Ey))
    #obj.setFill('orange')
    obj.draw(win)
    # ----------- Ліва грань ----------------------------------------------------
    obj = Polygon(Point(Ax, Ay), Point(Mx, My), Point(Ex, Ey), Point(Dx, Dy))
    obj.setFill('indigo')
    obj.draw(win)
    # ----------- Права грань ----------------------------------------------------
    obj = Polygon(Point(Bx, By), Point(Ix, Iy), Point(Fx, Fy), Point(Cx, Cy))
    #obj.setFill('yellow')
    obj.draw(win)
    return PrlpdWizRRR

#------------------------- алгоритм Брезенхема для растрофікації ліній ----------------------------------
def draw_line_pixel (x1, y1, x2, y2):
        dx = x2 - x1;  dy = y2 - y1
        sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
        sign_y = 1 if dy > 0 else -1 if dy < 0 else 0
        if dx < 0: dx = -dx
        if dy < 0: dy = -dy
        if dx > dy:
            pdx, pdy = sign_x, 0
            es, el = dy, dx
        else:
            pdx, pdy = 0, sign_y
            es, el = dx, dy
        x, y = x1, y1
        error, t = el / 2, 0

        obj = Point(x, y)
        obj.setFill('blue')
        obj.draw(win)

        while t < el:
            error -= es
            if error < 0:
                error += el
                x += sign_x
                y += sign_y
            else:
                x += pdx
                y += pdy
            t += 1
            obj = Point(x, y)
            obj.setFill('blue')
            obj.draw(win)
        return draw_line_pixel

#----------------- алгоритм Брезенхема для кольорової растрофікації ліній -----------------------------
def draw_line_pixel_color(x1, y1, x2, y2):
        red=1.0; green=1.0; blue=1.0
        dx = x2 - x1;  dy = y2 - y1
        sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
        sign_y = 1 if dy > 0 else -1 if dy < 0 else 0
        if dx < 0: dx = -dx
        if dy < 0: dy = -dy
        if dx > dy:
            pdx, pdy = sign_x, 0
            es, el = dy, dx
        else:
            pdx, pdy = 0, sign_y
            es, el = dx, dy
        x, y = x1, y1
        error, t = el / 2, 0

        obj = Point(x, y)
        obj.setFill('blue')
        obj.draw(win)

        while t < el:
            error -= es
            if error < 0:
                error += el
                x += sign_x
                y += sign_y
            else:
                x += pdx
                y += pdy
            t += 1
            obj = Point(x, y)
            red = red-0.001; green = green-0.001; blue = blue-0.001
            #print('red=', red)
            # перетворення RGB в шеснадцятирічний код з пакету matplotlib !!!
            col16 = colors.rgb2hex((red, green, blue))
            # інтеграція змінної до поля строки !!! СПОСІБ - Конкатенація строк
            obj.setFill(''+ col16)
            obj.draw(win)
        return draw_line_pixel_color

#------------------- функція побудови растрового паралелепіпеда -----------------------
def PrlpdWiz_Pixel(Prxy3):
    # ----------- дальня грань - (в проекції ліва) -------------------------------------
    Ax1 = Prxy3[0, 0]
    Ay1 = Prxy3[0, 1]
    Bx1 = Prxy3[1, 0]
    By1 = Prxy3[1, 1]
    Ix1 = Prxy3[2, 0]
    Iy1 = Prxy3[2, 1]
    Mx1 = Prxy3[3, 0]
    My1 = Prxy3[3, 1]
    # ----------- ближня грань - (в проекції права) -------------------------------------
    Dx1 = Prxy3[4, 0]
    Dy1 = Prxy3[4, 1]
    Cx1 = Prxy3[5, 0]
    Cy1 = Prxy3[5, 1]
    Fx1 = Prxy3[6, 0]
    Fy1 = Prxy3[6, 1]
    Ex1 = Prxy3[7, 0]
    Ey1 = Prxy3[7, 1]

    # ----------- дальня грань - (в проекції ліва) -------------------------------------
    draw_line_pixel(Ax1, Ay1, Bx1, By1)
    draw_line_pixel(Bx1, By1, Ix1, Iy1)
    draw_line_pixel(Ix1, Iy1, Mx1, My1)
    draw_line_pixel(Mx1, My1, Ax1, Ay1)
    # ----------- ближча грань - (в проекції права) -------------------------------------
    draw_line_pixel(Dx1, Dy1, Cx1, Cy1)
    draw_line_pixel(Cx1, Cy1, Fx1, Fy1)
    draw_line_pixel(Fx1, Fy1, Ex1, Ey1)
    draw_line_pixel(Ex1, Ey1, Dx1, Dy1)
    # ----------- верхеня грань - (в проекції верхня) -------------------------------------
    draw_line_pixel(Ax1, Ay1, Bx1, By1)
    draw_line_pixel(Bx1, By1, Cx1, Cy1)
    draw_line_pixel(Cx1, Cy1, Dx1, Dy1)
    draw_line_pixel(Dx1, Dy1, Ax1, Ay1)
    # ----------- верхеня грань - (в проекції верхня) -------------------------------------
    draw_line_pixel(Mx1, My1, Ix1, Iy1)
    draw_line_pixel(Ix1, Iy1, Fx1, Fy1)
    draw_line_pixel(Fx1, Fy1, Ex1, Ey1)
    draw_line_pixel(Ex1, Ey1, Mx1, My1)
    # ----------- ліва грань - (в проекції ближня) ----------------------------------------
    draw_line_pixel(Ax1, Ay1, Mx1, My1)
    draw_line_pixel(Mx1, My1, Ex1, Ey1)
    draw_line_pixel(Ex1, Ey1, Dx1, Dy1)
    draw_line_pixel(Dx1, Dy1, Ax1, Ay1)
    # ----------- права грань - (в проекції дальня) ----------------------------------------
    draw_line_pixel(Bx1, By1, Ix1, Iy1)
    draw_line_pixel(Ix1, Iy1, Fx1, Fy1)
    draw_line_pixel(Fx1, Fy1, Cx1, Cy1)
    draw_line_pixel(Cx1, Cy1, Bx1, By1)
    return PrlpdWiz_Pixel

#------------------- функція побудови растрового паралелепіпеда -----------------------
def PrlpdWiz_Pixel_color(Prxy3):
    # ----------- дальня грань - (в проекції ліва) -------------------------------------
    Ax1 = Prxy3[0, 0]
    Ay1 = Prxy3[0, 1]
    Bx1 = Prxy3[1, 0]
    By1 = Prxy3[1, 1]
    Ix1 = Prxy3[2, 0]
    Iy1 = Prxy3[2, 1]
    Mx1 = Prxy3[3, 0]
    My1 = Prxy3[3, 1]
    # ----------- ближня грань - (в проекції права) -------------------------------------
    Dx1 = Prxy3[4, 0]
    Dy1 = Prxy3[4, 1]
    Cx1 = Prxy3[5, 0]
    Cy1 = Prxy3[5, 1]
    Fx1 = Prxy3[6, 0]
    Fy1 = Prxy3[6, 1]
    Ex1 = Prxy3[7, 0]
    Ey1 = Prxy3[7, 1]

    # ----------- дальня грань - (в проекції ліва) -------------------------------------
    draw_line_pixel_color(Ax1, Ay1, Bx1, By1)
    draw_line_pixel_color(Bx1, By1, Ix1, Iy1)
    draw_line_pixel_color(Ix1, Iy1, Mx1, My1)
    draw_line_pixel_color(Mx1, My1, Ax1, Ay1)
    # ----------- ближча грань - (в проекції права) -------------------------------------
    draw_line_pixel_color(Dx1, Dy1, Cx1, Cy1)
    draw_line_pixel_color(Cx1, Cy1, Fx1, Fy1)
    draw_line_pixel_color(Fx1, Fy1, Ex1, Ey1)
    draw_line_pixel_color(Ex1, Ey1, Dx1, Dy1)
    # ----------- верхеня грань - (в проекції верхня) -------------------------------------
    draw_line_pixel_color(Ax1, Ay1, Bx1, By1)
    draw_line_pixel_color(Bx1, By1, Cx1, Cy1)
    draw_line_pixel_color(Cx1, Cy1, Dx1, Dy1)
    draw_line_pixel_color(Dx1, Dy1, Ax1, Ay1)
    # ----------- верхеня грань - (в проекції верхня) -------------------------------------
    draw_line_pixel_color(Mx1, My1, Ix1, Iy1)
    draw_line_pixel_color(Ix1, Iy1, Fx1, Fy1)
    draw_line_pixel_color(Fx1, Fy1, Ex1, Ey1)
    draw_line_pixel_color(Ex1, Ey1, Mx1, My1)
    # ----------- ліва грань - (в проекції ближня) ----------------------------------------
    draw_line_pixel_color(Ax1, Ay1, Mx1, My1)
    draw_line_pixel_color(Mx1, My1, Ex1, Ey1)
    draw_line_pixel_color(Ex1, Ey1, Dx1, Dy1)
    draw_line_pixel_color(Dx1, Dy1, Ax1, Ay1)
    # ----------- права грань - (в проекції дальня) ----------------------------------------
    draw_line_pixel_color(Bx1, By1, Ix1, Iy1)
    draw_line_pixel_color(Ix1, Iy1, Fx1, Fy1)
    draw_line_pixel_color(Fx1, Fy1, Cx1, Cy1)
    draw_line_pixel_color(Cx1, Cy1, Bx1, By1)
    return PrlpdWiz_Pixel_color

#------------------------ аксонометричний векторний МОНОХРОМ паралелепіпед -------------------
win = GraphWin("3-D  векторний МОНОХРОМ паралелепіпед аксонометрічна проекція на ХУ", xw, yw)
win.setBackground('white')
xw=600; yw=600; st=50; TetaG1=180; TetaG2=-90
l=(xw/2)-st; m=(yw/2)-st; n=m
Prlpd1=ShiftXYZ (Prlpd, l, m, n)
Prlpd2=dimetri (Prlpd1, TetaG1, TetaG2)
Prxy3=ProjectXY (Prlpd2)
PrlpdWiz(Prxy3)
win.getMouse()
win.close()

#--------------------------- аксонометричний векторний КОЛОР паралелепіпед -------------------
win = GraphWin("3-D  векторний КОЛОР паралелепіпед аксонометрічна проекція на ХУ", xw, yw)
win.setBackground('white')
xw=600; yw=600; st=50; TetaG1=180; TetaG2=-90
l=(xw/2)-st; m=(yw/2)-st; n=m;
Prlpd1=ShiftXYZ (Prlpd, l, m, n)
Prlpd2=dimetri (Prlpd1, TetaG1, TetaG2)
Prxy3=ProjectXY (Prlpd2)
PrlpdWizRRR(Prxy3)
win.getMouse()
win.close()

#------------------------------------- аксонометричний растровий паралелепіпед  -------------
win = GraphWin("3-D растровий паралелепіпеда аксонометрічна проекція на ХУ", xw, yw)
win.setBackground('white')
xw=600; yw=600; st=50; TetaG1=180; TetaG2=-90
l=(xw/2)-st; m=(yw/2)-st; n=m;
Prlpd1=ShiftXYZ (Prlpd, l, m, n)
Prlpd2=dimetri (Prlpd1, TetaG1, TetaG2)
Prxy3=ProjectXY (Prlpd2)
PrlpdWiz_Pixel(Prxy3)
win.getMouse()
win.close()

#--------------------------- кольоровий аксонометричний растровий паралелепіпед  -------------
win = GraphWin("3-D растровий паралелепіпеда аксонометрічна проекція на ХУ", xw, yw)
win.setBackground('white')
xw=600; yw=600; st=50; TetaG1=180; TetaG2=-90
l=(xw/2)-st; m=(yw/2)-st; n=m;
Prlpd1=ShiftXYZ (Prlpd, l, m, n)
Prlpd2=dimetri (Prlpd1, TetaG1, TetaG2)
Prxy3=ProjectXY (Prlpd2)
PrlpdWiz_Pixel_color(Prxy3)
win.getMouse()
win.close()