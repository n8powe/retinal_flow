#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monkey Retinal Flow project
3. retinocentric:
In this script we use the pupil position extracted in step 1 and calibration targets
position extracted in step 2 to calibrate eye position and compute images in retino-
centric coordinates.
"""

import time
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
dataPath = '../../retinal_flow_data'
eyeFile = 'eye_2022-11-08_19-54-09_eyePosition.csv'
sceneFile = 'scene_2022-11-08_19-54-09_targetPosition.csv'

eyeDat = pd.read_csv(dataPath+'/'+eyeFile, sep=',', names=['F',	'T', 'I', 'X', 'Y', 'D', 'C'], header=0, skiprows=0).values
sceneDat = pd.read_csv(dataPath+'/'+sceneFile, sep=',', names=['T', 'I', 'X', 'Y'], header=0, skiprows=0).values

# Find trigger times
eyeTrigOn, = np.where(eyeDat[1:, 2]-eyeDat[:-1, 2] > 0.5)
eyeTrigOff, = np.where(eyeDat[1:, 2]-eyeDat[:-1, 2] < -0.5)
sceneTrigOn, = np.where(sceneDat[1:, 1]-sceneDat[:-1, 1] > 0.5)
sceneTrigOff, = np.where(sceneDat[1:, 1]-sceneDat[:-1, 1] < -0.5)
eyeStart = eyeTrigOn[0]
eyeStop = eyeTrigOff[-1]
sceneStart = sceneTrigOn[0]
sceneStop = sceneTrigOff[-1]

plt.plot(eyeDat[eyeStart-10:eyeStop+10, 1]-eyeDat[eyeStart, 1], eyeDat[eyeStart-10:eyeStop+10, 2],
         sceneDat[sceneStart-10:sceneStop+10, 0]-sceneDat[sceneStart, 0], sceneDat[sceneStart-10:sceneStop+10, 1])
plt.xlabel('Time (msec)')
plt.ylabel('Trigger')
plt.title("Triggers")
plt.legend(['Eye', 'Scene'])
plt.show()


# Focus on the calibration task
eyeCalibStart = eyeTrigOn[1]
eyeCalibStop = eyeTrigOff[1]
sceneCalibStart = sceneTrigOn[1]
sceneCalibStop = sceneTrigOff[1]

# Plot eye at target positions as a function of time
plt.subplot(221)
plt.plot(eyeDat[eyeCalibStart:eyeCalibStop, 1]-eyeDat[eyeStart, 1], eyeDat[eyeCalibStart:eyeCalibStop, 3]),
plt.ylabel('Eye X')
plt.subplot(222)
plt.plot(eyeDat[eyeCalibStart:eyeCalibStop, 1]-eyeDat[eyeStart, 1], eyeDat[eyeCalibStart:eyeCalibStop, 4]),
plt.ylabel('Eye Y')
plt.subplot(223)
plt.plot(sceneDat[sceneCalibStart:sceneCalibStop, 0]-sceneDat[sceneStart, 0], sceneDat[sceneCalibStart:sceneCalibStop, 2])
plt.xlabel('Time (msec)')
plt.ylabel('Target X')
plt.subplot(224)
plt.plot(sceneDat[sceneCalibStart:sceneCalibStop, 0]-sceneDat[sceneStart, 0], sceneDat[sceneCalibStart:sceneCalibStop, 3])
plt.xlabel('Time (msec)')
plt.ylabel('Target Y')
plt.show()

# Plot target as a function of eye
resamplingTimes = np.arange(0, round(eyeDat[eyeCalibStop, 1]-eyeDat[eyeStart, 1]))
eyeXresampled = np.interp(resamplingTimes,
                          eyeDat[eyeCalibStart:eyeCalibStop, 1]-eyeDat[eyeStart, 1],
                          eyeDat[eyeCalibStart:eyeCalibStop, 3])
sceneXresampled = np.interp(resamplingTimes,
                          sceneDat[sceneCalibStart:sceneCalibStop, 1]-sceneDat[sceneStart, 0],
                          sceneDat[sceneCalibStart:sceneCalibStop, 3])

plt.subplot(121)
plt.plot(eyeXresampled, sceneXresampled, '.')
plt.xlabel('Eye X')
plt.ylabel('Target X')
# plt.subplot(122)
# plt.plot(eyeDat[eyeCalibStart:eyeCalibStop, 4], sceneDat[sceneCalibStart:sceneCalibStop, 3], '.')
# plt.xlabel('Eye Y')
# plt.ylabel('Target Y')
plt.show()