# -------  апроксимація растрового зобреження за МНК з отриманням векторної форми ----------------

'''

Функціонал скрипта:
побудова куба геометричним методом;
растеризація;
векторизація за МНК.

Package                      Version
---------------------------- -----------
pip                          23.1
munch                        2.5.0
numpy                        1.23.5

'''

from graphics import *
import numpy as np
import math as mt

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

#------------------------- алгоритм Брезенхема + MNK ----------------------------------
def MNK (x1, y1, x2, y2):
# -------- алгоритм Брезенхема для визначення параметрів матриць MNK ----------------
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
        obj.setFill('orange')
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
            obj.setFill('orange')
            obj.draw(win)
# -------- оголошення нульового та одиничного МНК масивів ------------------
        stopt=t
        print(stopt)
        Yin=np.zeros((stopt, 1))
        F=np.ones((stopt, 2))
        FX=np.ones((stopt, 2))

        print('Fvx=', F)
        print('Yivx=', Yin)
        # -------- алгоритм Брезенхема для заповнення  матриць MNK ------------------
        dx = x2 - x1
        dy = y2 - y1
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
        obj.setFill('orange')
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
# ---------------------- формування матриці F - MNK ----------------------------------
#             tt = t
            tt=t-1
            # print('tt=', tt)
            Yin[tt, 0] = float(y)
            F[tt, 1] = float(x)
            FX[tt, 1] = float(x)
            obj.setFill('orange')
            obj.draw(win)
# ----------------- корегування матрицы F для випадку координат констант ---------------
        for i in range (0, stopt):
             F[i, 1]=i
# ------------------------------- алгоритм - MNK --------------------------------------
        FT=F.T
        FFT = FT.dot(F)
        FFTI=np.linalg.inv(FFT)
        FFTIFT=FFTI.dot(FT)
        C=FFTIFT.dot(Yin)
        Yout=F.dot(C)
# ----------------------- контроль обчислень алгоритма - MNK --------------------------
        print('F=', F)
        print('Yin=', Yin)
        print('FT=', FT)
        print('FFT=', FFT)
        print('FFTI=', FFTI)
        print('FFTIFT=', FFTIFT)
        print('C=', C)
        print('Yin=', Yin)
        print('Yout=', Yout, 'Yin=', Yin)

# -------------------- побудова ліній за координатами МНК ---------------------------------

        for i in range (0, stopt):
             XMNK = FX[i, 1]
             YMNK = Yout[i, 0]
             obj = Point(XMNK, YMNK)
             obj.setFill('violet')
             obj.draw(win)

        return MNK

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
def PrlpdWiz_MNK(Prxy3):
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
    MNK(Ax1, Ay1, Bx1, By1)
    MNK(Bx1, By1, Ix1, Iy1)
    MNK(Ix1, Iy1, Mx1, My1)
    MNK(Mx1, My1, Ax1, Ay1)
    # ----------- ближча грань - (в проекції права) -------------------------------------
    MNK(Dx1, Dy1, Cx1, Cy1)
    MNK(Cx1, Cy1, Fx1, Fy1)
    MNK(Fx1, Fy1, Ex1, Ey1)
    MNK(Ex1, Ey1, Dx1, Dy1)
    # ----------- верхеня грань - (в проекції верхня) -------------------------------------
    MNK(Ax1, Ay1, Bx1, By1)
    MNK(Bx1, By1, Cx1, Cy1)
    MNK(Cx1, Cy1, Dx1, Dy1)
    MNK(Dx1, Dy1, Ax1, Ay1)
    # ----------- верхеня грань - (в проекції верхня) -------------------------------------
    MNK(Mx1, My1, Ix1, Iy1)
    MNK(Ix1, Iy1, Fx1, Fy1)
    MNK(Fx1, Fy1, Ex1, Ey1)
    MNK(Ex1, Ey1, Mx1, My1)
    # ----------- ліва грань - (в проекції ближня) ----------------------------------------
    MNK(Ax1, Ay1, Mx1, My1)
    MNK(Mx1, My1, Ex1, Ey1)
    MNK(Ex1, Ey1, Dx1, Dy1)
    MNK(Dx1, Dy1, Ax1, Ay1)
    # ----------- права грань - (в проекції дальня) ----------------------------------------
    MNK(Bx1, By1, Ix1, Iy1)
    MNK(Ix1, Iy1, Fx1, Fy1)
    MNK(Fx1, Fy1, Cx1, Cy1)
    MNK(Cx1, Cy1, Bx1, By1)
    return PrlpdWiz_MNK

# ----------------------------------------- головні виклики --------------------------------

if __name__ == '__main__':
    # ----------------------------------- початкові параметри ------------------------------
    xw = 600
    yw = 600
    st = 50
    TetaG1 = 180
    TetaG2 = -90
    l = (xw / 2) - st
    m = (yw / 2) - st
    n = m

    # ------------------------ просторовий куб -------------------
    win = GraphWin("3-D  векторний МОНОХРОМ паралелепіпед аксонометрічна проекція на ХУ", xw, yw)
    win.setBackground('white')
    Prlpd1 = ShiftXYZ(Prlpd, l, m, n)
    Prlpd2 = dimetri(Prlpd1, TetaG1, TetaG2)
    Prxy3 = ProjectXY(Prlpd2)
    PrlpdWiz(Prxy3)
    win.getMouse()
    win.close()

    # ---------------------------- растеризація  -----------------
    win = GraphWin("3-D растровий паралелепіпеда аксонометрічна проекція на ХУ", xw, yw)
    win.setBackground('white')
    Prlpd1 = ShiftXYZ(Prlpd, l, m, n)
    Prlpd2 = dimetri(Prlpd1, TetaG1, TetaG2)
    Prxy3 = ProjectXY(Prlpd2)
    PrlpdWiz_Pixel(Prxy3) # растеризація
    win.getMouse()
    win.close()

    # --------------------------- векторизація за МНК -------------
    win = GraphWin("3-D растровий паралелепіпеда аксонометрічна проекція на ХУ", xw, yw)
    win.setBackground('white')
    Prlpd1 = ShiftXYZ(Prlpd, l, m, n)
    Prlpd2 = dimetri(Prlpd1, TetaG1, TetaG2)
    Prxy3 = ProjectXY(Prlpd2)
    PrlpdWiz_MNK(Prxy3)  # векторизація
    win.getMouse()
    win.close()

