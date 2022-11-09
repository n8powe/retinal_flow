#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monkey Retinal Flow project
2. Process scene:
This script reads a scene video. It extracts the position of the target
during the calibration task
"""

import time
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Scene video name and path
dataPath = '../../retinal_flow_data/'
fileName = 'eye_2022-11-08_19-54-09'

# Open video and trigger files
vidIn = cv2.VideoCapture('%s/%s_processed.avi' % (dataPath, fileName))
data = pd.read_csv('%s/%s.txt' % (dataPath, fileName), sep=',', names=['frameTimes', 'trigger'], header=0,
                   skiprows=0).values

imageSize = (int(vidIn.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidIn.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter('%s/%s_trimmed.mp4' % (dataPath, fileName), fourcc, 30, imageSize)

trigOn, = np.where(data[1:, 1] - data[:-1, 1] > 0.5)
trigOff, = np.where(data[1:, 1] - data[:-1, 1] < -0.5)
trigDur = trigOff - trigOn
taskStart = trigOn[trigDur > 80]
taskStop = trigOff[trigDur > 80]

# if not os.path.exists('%s/%s_charuco' % (dataPath, fileName)):
#     os.mkdir('%s/%s_charuco' % (dataPath, fileName))

gridSize = (14, 7)
boxWidthPix = 8 * 16
cornerThreshold = 4
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(gridSize[0], gridSize[1], boxWidthPix, 0.75 * boxWidthPix, dictionary)


print("Detect charuco board")
allCharucoCorners = []
allCharucoIds = []
vidIn.set(cv2.CAP_PROP_POS_FRAMES, taskStart[0]+1)
for ff in range(taskStart[0]+1,taskStop[0]):
    ret, frame = vidIn.read()
    # cv2.imwrite('%s/%s_charuco/%s_charuco_%d.png' % (dataPath,fileName,fileName,ff) , frame)
    writer.write(frame)

    arucoCorners, arucoIds, arucoRejected = cv2.aruco.detectMarkers(frame,dictionary)
    cv2.aruco.drawDetectedMarkers(frame,arucoCorners, arucoIds, borderColor=(0,0,255));

    if len(arucoCorners)>0:
        charucoNumber, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(arucoCorners, arucoIds, frame, board);

        if charucoNumber>cornerThreshold:
            allCharucoCorners.append(charucoCorners)
            allCharucoIds.append(charucoIds)

    cv2.imshow('picture', frame)
    cv2.waitKey(1)

writer.release()
cv2.waitKey(-1)



#
# # retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(allCharucoCorners, allCharucoIds, board, imageSize, None, None)
#
# # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize)
#
#
# # vidIn.set(cv2.CAP_PROP_POS_FRAMES, taskStart[0]+1)
# # for ff in range(taskStart[0]+1,taskStop[0]):
# #     ret, frame = vidIn.read()
# #     # cv2.imwrite('%s/%s_charuco/%s_charuco_%d.png' % (dataPath,fileName,fileName,ff) , frame)
#
# #     undistorted = cv2.undistort(frame, cameraMatrix, distCoeffs, None, newcameramtx)
#
# #     # cv2.imshow('picture', undistorted)
# #     # cv2.waitKey(1)
#
#
# print("Detect targets")
# # targetPosition = pd.DataFrame(columns = ['T','X','Y'])
# data2 = pd.DataFrame(data=data)
# data2 = data2.assign(tgX=np.full(data2.shape[0], np.nan))
# data2 = data2.assign(tgY=np.full(data2.shape[0], np.nan))
#
# marginX = 0  # int(np.min(allCharucoCorners[0][:,0,0]))
# marginY = 0  # int(np.min(allCharucoCorners[0][:,0,1]))
#
# vidIn.set(cv2.CAP_PROP_POS_FRAMES, taskStart[1] + 1)
# for ff in range(taskStart[1] + 1, taskStop[1]):
#     ret, frame = vidIn.read()
#     # cv2.imwrite('%s/%s_charuco/%s_charuco_%d.png' % (dataPath,fileName,fileName,ff) , frame)
#
#     gray = frame[marginX:frame.shape[0], marginY:frame.shape[1]]
#     gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
#
#     # undistorted = cv2.undistort(gray, cameraMatrix, distCoeffs, None, newcameramtx)
#     gray = cv2.medianBlur(gray, 5)
#
#     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1,
#                                param1=50, param2=10,
#                                minRadius=5, maxRadius=20)
#
#     if circles is not None:
#         tgX = int(np.mean(circles[:, :, 0])) + marginX
#         tgY = int(np.mean(circles[:, :, 1])) + marginY
#         cv2.circle(gray, (tgX, tgY), 10, (255, 255, 255), 3)
#
#         data2['tgX'].loc[ff] = tgX
#         data2['tgY'].loc[ff] = tgY
#         # data2.assign(tgX=np.full(data2.shape[0],np.nan))
#         # targetPosition = targetPosition.append({'T':ff,'X': tgX,'Y': tgY},ignore_index=True)
#         # loc[len(eyePosition.index)] = [result_2d["ellipse"]['center'][0],result_2
#
#     cv2.imshow("detected circles", gray)
#     cv2.waitKey(-1)
#
# # save the dataframe as a csv file
# # targetPosition.to_csv('%s/%s_targetPosition.csv' % (dataPath,fileName))
# data2.to_csv('%s/%s_targetPosition.csv' % (dataPath, fileName))

vidIn.release()

cv2.waitKey(-1)
cv2.destroyAllWindows()