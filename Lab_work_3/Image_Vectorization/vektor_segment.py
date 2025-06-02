# --- приклад векторизація растрового зображення для визначення контуру -------------------

'''
Функціонал:
кольорова корекція растрового зображення;
векторизація контуру об'єкта в растровому зображенні;

Package                      Version
---------------------------- -----------
opencv-python                3.4.18.65
pip                          23.1
Pillow                       9.4.0

'''



from PIL import Image
from pylab import *
import cv2

def Сolor_Vectorization (FileIm):
    # -----------  кольорова векторізація з внутрішнім заповненням ---------------
    # зчитування піксельного зображення в масив
    im = array(Image.open(FileIm).convert('L'))
    # створити нову фігуру
    figure()
    # відобразити контур
    contour(im, origin='image')
    axis('equal')
    show()
    return

def Monochrome_Vectorization (FileIm):
    # -----------  монохромна векторізація без внутрішнього заповнення ---------------
    im = array(Image.open(FileIm).convert('L'))
    # створити нову фігуру
    figure()
    # відобразити контур монохром без внутрішнього заповнення
    contour(im, levels=[245], colors='black', origin='image')
    axis('equal')
    show()
    return

def Canny_Vectorization (FileIm):
    img = cv2.imread(FileIm)
    # ------------------------ векторізація з Canny ---------------------------------
    blur1 = cv2.Canny(img, 100, 200)
    # ----------------------- відображення результату --------------------------------
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur1), plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()
    return

if __name__ == '__main__':
    # ------------------ кольорова векторізація з внутрішнім заповненням ------------------
    Сolor_Vectorization('start.jpg')
    Сolor_Vectorization('sentinel_2023.jpg')
    Сolor_Vectorization('segmentKmeans.jpg')
    # Сolor_Vectorization ('rez_equalize_hist.jpg')         - монохром не підпорядковано алгоритмам кольоровій корекції

    # ---------------- монохромна векторізація без внутрішнього заповнення ----------------
    Monochrome_Vectorization('start.jpg')
    Monochrome_Vectorization('sentinel_2023.jpg')
    # Monochrome_Vectorization ('segmentKmeans.jpg')        - властивості кольору не відповідають заданому порогу рішення
    # Monochrome_Vectorization ('rez_equalize_hist.jpg')    - монохром не підпорядковано алгоритмам кольоровій корекції

    # ---------------- монохромна векторізація без внутрішнього заповнення ----------------
    Canny_Vectorization('start.jpg')
    Canny_Vectorization('sentinel_2023.jpg')
    Canny_Vectorization('segmentKmeans.jpg')
    # Canny_Vectorization ('rez_equalize_hist.jpg')         - монохром не підпорядковано алгоритмам кольоровій корекції