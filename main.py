import cv2
import matplotlib.pyplot as plt
import os
#import pytesseract
from numpy.typing import NDArray


def load_image(img_path):
    try:
        number_plate_img = cv2.imread(img_path)
        number_plate_img = cv2.cvtColor(number_plate_img, cv2.COLOR_BGR2GRAY)
        return number_plate_img
    except (IOError, FileNotFoundError) as e:
        print(e)
        return None


def moto_plate_extract(image,
                       num_pl_haar_cascade,
                       rus16_haar_cascade):
    cv2.imshow('test', image)
    cv2.waitKey(0)
    cascade_np_rects = num_pl_haar_cascade.detectMultiScale(image,
                                                            scaleFactor=1.1,
                                                            minNeighbors=5)
    cascade_rus16_rects = rus16_haar_cascade.detectMultiScale(image,
                                                              scaleFactor=1.1,
                                                              minNeighbors=5)
    for x, y, w, h in cascade_rus16_rects: # + cascade_np_rects:
        moto_plate_image = image[y+15:y+h-10, x+15:x+w-20]
    return moto_plate_image


def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized_image


def main():
    print(cv2.__version__)
    file_path = os.getcwd()+ os.path.join('\test', '\simple',
                             'nomera-na-mototsikl-750x350.jpg')
    file_path = os.getcwd() + '/test/simple/nomera-na-mototsikl-750x350.jpg'
    file_path = os.getcwd() + '/test/simple/car2.jpg'
    number_plate_rgb = load_image(file_path)
    # cv2.imshow('test', number_plate_rgb)
    # cv2.waitKey(0)
    # plt.imshow(number_plate_rgb)
    # plt.show()
    rus16_cascade_name = 'cascades/haarcascade_license_plate_rus_16stages.xml'
    num_pl_cascade_name = 'cascades/haarcascade_russian_plate_number.xml'
    print(rus16_cascade_name)
    rus16_haar_cascade = cv2.CascadeClassifier()
    num_pl_haar_cascade = cv2.CascadeClassifier(num_pl_cascade_name)
    print(rus16_haar_cascade.empty())
    # if rus16_haar_cascade.empty() or num_pl_haar_cascade.empty():
    #     print('Cascade wasn\'t load')
    #     exit(0)
    print(cv2.samples.findFile(rus16_cascade_name))
    if not rus16_haar_cascade.load(cv2.samples.findFile(rus16_cascade_name)):
        print('--(!)Error loading rus16 cascade')
        exit(0)
    print(rus16_haar_cascade.empty())
    if not num_pl_haar_cascade.load(cv2.samples.findFile(num_pl_cascade_name)):
        print('--(!)Error loading number plate cascade')
        exit(0)

    moto_plate_extract_img = moto_plate_extract(number_plate_rgb,
                                                num_pl_haar_cascade,
                                                rus16_haar_cascade)
    moto_plate_extract_img = enlarge_img(moto_plate_extract_img, 150)
    plt.imshow(moto_plate_extract_img)
    plt.show()


if __name__ == '__main__':
    main()
