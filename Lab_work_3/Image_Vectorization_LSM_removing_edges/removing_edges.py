#--------- приклад побудови векторного паралелепіпеда з видаленям невидимих граней ------------

'''

Функціонал скрипта:
побудова куба геометричним методом;
растеризація;
векторизація за МНК;
видалення невидимих ліній та граней - Алгоритм плаваючого обрію.

Package                      Version
---------------------------- -----------
pip                          23.1
munch                        2.5.0
numpy                        1.23.5

'''

from graphics import *
import numpy as np
import math as mt

#---------------------------------- координати паралелепіпеда ---------------------------------

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
    TetaR1=(3/14*TetaG1)/180; TetaR2=(3/14*TetaG2)/180
    f1 = np.array([[mt.cos(TetaR1), 0 , -mt.sin(TetaR1), 0], [0, 1, 0, 0],  [mt.sin(TetaR1), 0, mt.cos(TetaR1), 1], [0, 0, 0, 0],])
    ft1 = f1.T
    Prxy1 = Figure.dot(ft1)
    f2 = np.array([ [1, 0, 0, 0], [0, mt.cos(TetaR2), mt.sin(TetaR2), 0], [0, -mt.sin(TetaR2),  mt.cos(TetaR2), 0], [0, 0, 0, 1] ])
    ft2=f2.T
    Prxy2 = Prxy1.dot(ft2)
    print('dimetri')
    print(Prxy2)
    return Prxy2

#-------------------- функція побудови векторного паралелепіпеда - МОНОХРОМ -------------
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

#-------------------- функція побудови векторного паралелепіпеда -КОЛІР ------------
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

#--------- функція побудови векторного паралелепіпеда з видаленям невидимих граней ------------
def PrlpdWizReal_G(PrxyDIM, Prxy, Xmax, Ymax, Zmax):
    # ----------- Алгоритм плаваючого обрію --------------------------------------

    # ----------- аксонометрична матриця без проекції -------------------------------

    AAx = PrxyDIM[0, 0];  AAy = PrxyDIM[0, 1]; AAz = PrxyDIM[0, 2]
    BBx = PrxyDIM[1, 0];  BBy = PrxyDIM[1, 1]; BBz = PrxyDIM[1, 2]
    IIx = PrxyDIM[2, 0];  IIy = PrxyDIM[2, 1]; IIz = PrxyDIM[2, 2]
    MMx = PrxyDIM[3, 0];  MMy = PrxyDIM[3, 1]; MMz = PrxyDIM[3, 2]

    DDx = PrxyDIM[4, 0];  DDy = PrxyDIM[4, 1]; DDz = PrxyDIM[4, 2]
    CCx = PrxyDIM[5, 0];  CCy = PrxyDIM[5, 1]; CCz = PrxyDIM[5, 2]
    FFx = PrxyDIM[6, 0];  FFy = PrxyDIM[6, 1]; FFz = PrxyDIM[6, 2]
    EEx = PrxyDIM[7, 0];  EEy = PrxyDIM[7, 1]; EEz = PrxyDIM[7, 2]
    # -------- УВАГА !!! аналізу підлягає оригінальна аксонометрична матриця без проекції ----------------
    # ------------------------Умови для кутів аксонометрії 0-180 грд -------------------------------------
    # -------------------------------------F-T--------------------------------------
    if (abs(AAz-Zmax) > abs(DDz-Zmax)) and (abs(BBz-Zmax) > abs(CCz-Zmax))  \
            and (abs(IIz-Zmax) > abs(FFz-Zmax)) and (abs(MMz-Zmax) > abs(EEz-Zmax)):
        FlagF = 1
    else:
        FlagF = 2
    print('FlagF=', FlagF)
    # -------------------------------------L-R--------------------------------------
    if (abs(DDx - Xmax) > abs(CCx - Xmax)) and (abs(AAx - Xmax) > abs(BBx - Xmax)) \
            and (abs(MMx - Xmax) > abs(IIx - Xmax)) and (abs(EEx - Xmax) > abs(FFx - Xmax)):
        FlagR = 1
    else:
        FlagR = 2
    print('FlagR=', FlagR)
    # -------------------------------------P-D--------------------------------------
    if (abs(AAy - Ymax) > abs(MMy - Ymax)) and (abs(BBy - Ymax) > abs(IIy - Ymax)) \
            and (abs(CCy - Ymax) > abs(FFy - Ymax)) and (abs(DDy - Ymax) > abs(EEy - Ymax)):
        FlagP = 1
    else:
        FlagP = 2
    print('FlagP=', FlagP)
    # -------------------------------------------------------------------------------
    Ax = Prxy[0, 0];  Ay = Prxy[0, 1]
    Bx = Prxy[1, 0];  By = Prxy[1, 1]
    Ix = Prxy[2, 0];  Iy = Prxy[2, 1]
    Mx = Prxy[3, 0];  My = Prxy[3, 1]

    Dx = Prxy[4, 0];  Dy = Prxy[4, 1]
    Cx = Prxy[5, 0];  Cy = Prxy[5, 1]
    Fx = Prxy[6, 0];  Fy = Prxy[6, 1]
    Ex = Prxy[7, 0];  Ey = Prxy[7, 1]

    # ----------- Ліва грань ----------------------------------------------------
    obj = Polygon(Point(Ax, Ay), Point(Mx, My), Point(Ex, Ey), Point(Dx, Dy))
    if FlagR == 2:
        obj.setFill('indigo')
        obj.draw(win)
    # ----------- Права грань ----------------------------------------------------
    obj = Polygon(Point(Bx, By), Point(Ix, Iy), Point(Fx, Fy), Point(Cx, Cy))
    if FlagR == 1:
        obj.setFill('indigo')
        obj.draw(win)
    # ----------- Верхня грань ----------------------------------------------------
    obj = Polygon(Point(Ax, Ay), Point(Bx, By), Point(Cx, Cy), Point(Dx, Dy))
    if FlagP == 1:
        obj.setFill('violet')
        obj.draw(win)
    # ----------- Нижня грань ----------------------------------------------------
    obj = Polygon(Point(Mx, My), Point(Ix, Iy), Point(Fx, Fy), Point(Ex, Ey))
    if FlagP == 2:
        obj.setFill('orange')
        obj.draw(win)
    #----------- Тильна грань ----------------------------------------------------
    obj = Polygon(Point(Ax, Ay), Point(Bx, By), Point(Ix, Iy), Point(Mx, My))
    if FlagF==2:
        obj.setFill('blue')
        obj.draw(win)
    # ----------- Фронтальна грань ------------------------------------------------
    obj = Polygon(Point(Dx, Dy), Point(Cx, Cy), Point(Fx, Fy), Point(Ex, Ey))
    if FlagF==1:
        obj.setFill('green')
        obj.draw(win)

    return PrlpdWizReal_G

#------------------------ аксонометричний векторний МОНОХРОМ паралелепіпед -------------------
win = GraphWin("3-D  векторний МОНОХРОМ паралелепіпед аксонометрічна проекція на ХУ", xw, yw)
win.setBackground('white')
xw=600; yw=600; st=50; TetaG1=180; TetaG2=-90
l=(xw/2)-st; m=(yw/2)-st; n=m;
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

win = GraphWin("3-D  векторний МОНОХРОМ паралелепіпед аксонометрічна проекція на ХУ", xw, yw)
win.setBackground('white')
xw=600; yw=600; st=50; TetaG1=160; TetaG2=40
l=(xw/2)-st; m=(yw/2)-st; n=m
Prlpd1=ShiftXYZ (Prlpd, l, m, n)
Prlpd2=dimetri (Prlpd1, TetaG1, TetaG2)
Prxy3=ProjectXY (Prlpd2)
PrlpdWiz(Prxy3)
win.getMouse()
win.close()

#--------- аксонометричний векторний КОЛОР паралелепіпед з видаленням невидимих граней---------
win = GraphWin("3-D  аксонометричний векторний КОЛОР паралелепіпед з видаленням невидимих граней", xw, yw)
win.setBackground('white')
xw=600; yw=600; st=50; TetaG1=160; TetaG2=40
l=(xw/2)-st; m=(yw/2)-st; n=m
Prlpd1=ShiftXYZ (Prlpd, l, m, n)
Prlpd2=dimetri (Prlpd1, TetaG1, TetaG2)
Prxy3=ProjectXY (Prlpd2)
PrlpdWizReal_G(Prlpd2, Prxy3, (xw*2), (yw*2), (yw*2))
win.getMouse()
win.close()

#--------- аксонометричний векторний КОЛОР паралелепіпед з видаленням невидимих граней---------
win = GraphWin("3-D  аксонометричний векторний КОЛОР паралелепіпед з видаленням невидимих граней", xw, yw)
win.setBackground('white')
xw=600; yw=600; st=50; TetaG1=180; TetaG2=-90
l=(xw/2)-st; m=(yw/2)-st; n=m
Prlpd1=ShiftXYZ (Prlpd, l, m, n)
Prlpd2=dimetri (Prlpd1, TetaG1, TetaG2)
Prxy3=ProjectXY (Prlpd2)
PrlpdWizReal_G(Prlpd2, Prxy3, (xw*2), (yw*2), (yw*2))
win.getMouse()
win.close()

