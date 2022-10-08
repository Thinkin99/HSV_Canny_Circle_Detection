from __future__ import division
import numpy as np
import pyrealsense2 as rs
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

def nothing(*arg):
        pass
 

para = (0, 127, 149, 255, 255, 255, 0, 50)
# lowHue lowSat lowVal highHue highSat highVal minRadius maxRadius
cv2.namedWindow('Trackbar')
cv2.resizeWindow('Trackbar', 500, 400)
cv2.createTrackbar('lowHue', 'Trackbar', para[0], 255, nothing)
cv2.createTrackbar('lowSat', 'Trackbar', para[1], 255, nothing)
cv2.createTrackbar('lowVal', 'Trackbar', para[2], 255, nothing)
cv2.createTrackbar('highHue', 'Trackbar', para[3], 255, nothing)
cv2.createTrackbar('highSat', 'Trackbar', para[4], 255, nothing)
cv2.createTrackbar('highVal', 'Trackbar', para[5], 255, nothing)
cv2.createTrackbar('minRadius', 'Trackbar', para[6], 500, nothing)
cv2.createTrackbar('maxRadius', 'Trackbar', para[7], 500, nothing)

def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

    ############### 相机参数的获取 #######################
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile(
    ).intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
    depth_image_3d = np.dstack(
        (depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图

    # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame
pipeline = rs.pipeline()  # 定义流程pipeline
config = rs.config()  # 定义配置config
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)  # 流程开始
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)


while True:
    # Get HSV values from the GUI sliders.
    intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images()  # 获取对齐的图像与相机内参
    if not depth_image.any() or not color_image.any():
        continue
    lowHue = cv2.getTrackbarPos('lowHue', 'Trackbar')
    lowSat = cv2.getTrackbarPos('lowSat', 'Trackbar')
    lowVal = cv2.getTrackbarPos('lowVal', 'Trackbar')
    highHue = cv2.getTrackbarPos('highHue', 'Trackbar')
    highSat = cv2.getTrackbarPos('highSat', 'Trackbar')
    highVal = cv2.getTrackbarPos('highVal', 'Trackbar')
    minRadius = cv2.getTrackbarPos('minRadius', 'Trackbar')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'Trackbar')
    print("para is ", [lowHue, lowSat, lowVal, highHue, highSat, highVal, minRadius, maxRadius])
    frame=color_image
    # Show the original image.
    cv2.namedWindow('frame',0)
    cv2.imshow('frame', frame)
    # Blur methods available, comment or uncomment to try different blur methods.
    frame = cv2.medianBlur(frame, 9)
    # Convert the frame to HSV colour model.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV values to define a colour range.
    colorLow = np.array([lowHue,lowSat,lowVal])
    colorHigh = np.array([highHue,highSat,highVal])
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
    result = cv2.bitwise_and(frame, frame, mask = mask)
    # Show final output image
    cv2.namedWindow('afterHSVmask',0)
    cv2.imshow('afterHSVmask', result)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    imgray = cv2.Canny(result, 600, 100, 3)  # Canny
    cv2.namedWindow('canny',0)
    cv2.imshow('canny',imgray)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
    list_xy=[]
    for cnt in contours:
        if len(cnt) > 50:
            ell = cv2.fitEllipse(cnt)  # 拟合椭圆 ellipse = [ center(x, y) , long short (a, b), angle ]
            a = ell[1][0]
            b = ell[1][1]
            x = int(ell[0][0])
            y = int(ell[0][1])
            if (b / a) < 1.2 and a > minRadius and b > minRadius and a < maxRadius and b < maxRadius:
                frame = cv2.ellipse(frame, ell, (0, 0, 200), 2)
                cv2.circle(frame, (x, y), 2, (255, 255, 255), 3)
                cv2.putText(frame, str((x, y,(a+b)//2)), (x + 20, y + 10), 0, 1,
                            [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                dis = aligned_depth_frame.get_distance(x, y)
                if dis == 0:
                    dis = aligned_depth_frame.get_distance(100, 100)#如果检测不到距离，就取（100，100）像素点的距离.
                camera_xyz = rs.rs2_deproject_pixel_to_point(
                    depth_intrin, (x, y), dis)  # 计算相机坐标系的xyz
                camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
                camera_xyz = camera_xyz.tolist()
                cv2.putText(frame, str(camera_xyz), (x - 50, y + 50), 0, 1,
                            [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)  # 标出坐标
    cv2.namedWindow("circle_detect",0)
    cv2.imshow("circle_detect", frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()