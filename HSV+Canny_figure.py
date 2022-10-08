import numpy as np
import cv2


def nothing(*arg):
    pass


para = (0, 127, 149, 255, 255, 255, 0, 50)
# lowHue lowSat lowVal highHue highSat highVal minRadius maxRadius
cv2.namedWindow('Trackbar')
cv2.resizeWindow('Trackbar', 400, 400)
cv2.createTrackbar('lowHue', 'Trackbar', para[0], 255, nothing)
cv2.createTrackbar('lowSat', 'Trackbar', para[1], 255, nothing)
cv2.createTrackbar('lowVal', 'Trackbar', para[2], 255, nothing)
cv2.createTrackbar('highHue', 'Trackbar', para[3], 255, nothing)
cv2.createTrackbar('highSat', 'Trackbar', para[4], 255, nothing)
cv2.createTrackbar('highVal', 'Trackbar', para[5], 255, nothing)
cv2.createTrackbar('minRadius', 'Trackbar', para[6], 500, nothing)
cv2.createTrackbar('maxRadius', 'Trackbar', para[7], 500, nothing)

while True:
    frame = cv2.imread("./c1.jpg")
    lowHue = cv2.getTrackbarPos('lowHue', 'Trackbar')
    lowSat = cv2.getTrackbarPos('lowSat', 'Trackbar')
    lowVal = cv2.getTrackbarPos('lowVal', 'Trackbar')
    highHue = cv2.getTrackbarPos('highHue', 'Trackbar')
    highSat = cv2.getTrackbarPos('highSat', 'Trackbar')
    highVal = cv2.getTrackbarPos('highVal', 'Trackbar')
    minRadius = cv2.getTrackbarPos('minRadius', 'Trackbar')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'Trackbar')
    print("para is ",[lowHue, lowSat, lowVal, highHue, highSat, highVal, minRadius, maxRadius])
    # Show the original image.
    cv2.namedWindow('frame', 0)
    cv2.imshow('frame', frame)
    # Blur methods available, comment or uncomment to try different blur methods.
    frame = cv2.medianBlur(frame, 5)
    # Convert the frame to HSV colour model.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV values to define a colour range.
    colorLow = np.array([lowHue, lowSat, lowVal])
    colorHigh = np.array([highHue, highSat, highVal])
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    # Show final output image
    cv2.namedWindow('afterHSVmask', 0)
    cv2.imshow('afterHSVmask', result)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    imgray = cv2.Canny(result, 600, 100, 3)  # Canny
    cv2.namedWindow('canny', 0)
    cv2.imshow('canny', imgray)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
    for cnt in contours:
        if len(cnt) > 50:
            ell = cv2.fitEllipse(cnt)  # 拟合椭圆 ell = [ center(x, y) , long short (a, b), angle ]
            a = ell[1][0]  # long
            b = ell[1][1]  # short
            x = int(ell[0][0])
            y = int(ell[0][1])
            if (b / a) < 1.2 and a > minRadius and b > minRadius and a < maxRadius and b < maxRadius:
                frame = cv2.ellipse(frame, ell, (0, 0, 200), 2)
                cv2.circle(frame, (x, y), 2, (255, 255, 255), 3)
                cv2.putText(frame, str((x, y, (a + b) // 2)), (x + 20, y + 10), 0, 1,
                            [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    cv2.namedWindow("circle_detect", 0)
    cv2.imshow("circle_detect", frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
