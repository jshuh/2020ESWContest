import cv2
import numpy as np
import dlib
import time
from scipy import signal

import psutil
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import playsound
import argparse
import imutils
import random
import psutil


WINDOW_TITLE = 'Pulse Observer'
BUFFER_MAX_SIZE = 500   
#MAX_VALUES_TO_GRAPH = 50  
MAX_VALUES_TO_GRAPH = 70  
MIN_HZ = 0.83        
MAX_HZ = 3.33        
MIN_FRAMES = 100     
DEBUG_MODE = False


#EYE VALUE ==============================================
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0
ITER    = 0
ALARM_ON = False
PREDICTOR_PATH = "d://68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def butterworth_filter(data, low, high, sample_rate, order=5):
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)

def get_forehead_roi(face_points):
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)
    min_x = int(points[21, 0])
    min_y = int(min(points[21, 1], points[22, 1]))
    max_x = int(points[22, 0])
    max_y = int(max(points[21, 1], points[22, 1]))
    left = min_x
    right = max_x
    top = min_y - (max_x - min_x)
    bottom = max_y * 0.98
    return int(left), int(right), int(top), int(bottom)

def get_nose_roi(face_points):
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)

    min_x = int(points[36, 0])
    min_y = int(points[28, 1])
    max_x = int(points[45, 0])
    max_y = int(points[33, 1])
    left = min_x
    right = max_x
    top = min_y + (min_y * 0.02)
    bottom = max_y + (max_y * 0.02)
    return int(left), int(right), int(top), int(bottom)


def get_full_roi(face_points):
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)
        
    min_x = int(np.min(points[17:47, 0]))
    min_y = int(np.min(points[17:47, 1]))
    max_x = int(np.max(points[17:47, 0]))
    max_y = int(np.max(points[17:47, 1]))

    center_x = min_x + (max_x - min_x) / 2
    left = min_x + int((center_x - min_x) * 0.15)
    right = max_x - int((max_x - center_x) * 0.15)
    top = int(min_y * 0.88)
    bottom = max_y
    return int(left), int(right), int(top), int(bottom)


def sliding_window_demean(signal_values, num_windows):
    window_size = int(round(len(signal_values) / num_windows))
    demeaned = np.zeros(signal_values.shape)
    for i in range(0, len(signal_values), window_size):
        if i + window_size > len(signal_values):
            window_size = len(signal_values) - i
        curr_slice = signal_values[i: i + window_size]
        if DEBUG_MODE and curr_slice.size == 0:
            print(  'Empty Slice: size={0}, i={1}, window_size={2}'.format(signal_values.size, i, window_size) )
            print(  curr_slice )
        demeaned[i:i + window_size] = curr_slice - np.mean(curr_slice)
    return demeaned


# Averages the green values for two arrays of pixels
def get_avg(roi1, roi2):
    roi1_green = roi1[:, :, 1]
    roi2_green = roi2[:, :, 1]
    avg = (np.mean(roi1_green) + np.mean(roi2_green)) / 2.0
    return avg


def get_max_abs(lst):
    return max(max(lst), -min(lst))


def draw_graph(signal_values, graph_width, graph_height):
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    scale_factor_x = float(graph_width) / MAX_VALUES_TO_GRAPH

    max_abs = get_max_abs(signal_values)
#    scale_factor_y = (float(graph_height) / 2.0) / max_abs
    scale_factor_y = (float(graph_height) / 2.0) / max_abs

    midpoint_y = graph_height / 2
    for i in range(0, len(signal_values) - 1):
        curr_x = int(i * scale_factor_x)
        curr_y = int(midpoint_y + signal_values[i] * scale_factor_y)
        next_x = int((i + 1) * scale_factor_x)
        next_y = int(midpoint_y + signal_values[i + 1] * scale_factor_y)
        cv2.line(graph, (curr_x, curr_y), (next_x, next_y), color=(0, 255, 0), thickness=1)
    return graph

def draw_graph2(signal_values, graph_width, graph_height):
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    scale_factor_x = float(graph_width) / MAX_VALUES_TO_GRAPH

    max_abs = get_max_abs(signal_values)
#    scale_factor_y = (float(graph_height) / 2.0) / max_abs
    scale_factor_y = 600 # (float(graph_height) / 2.0) / max_abs

    midpoint_y = -60 #graph_height / 2
    for i in range(0, len(signal_values) - 1):
        curr_x = int(i * scale_factor_x)
        curr_y = int(midpoint_y + signal_values[i] * scale_factor_y)
        next_x = int((i + 1) * scale_factor_x)
        next_y = int(midpoint_y + signal_values[i + 1] * scale_factor_y)
        cv2.line(graph, (curr_x, curr_y), (next_x, next_y), color=(0, 20, 255), thickness=2)
    return graph

def draw_bpm(ear_str,bpm_str, bpm_width, bpm_height):
    bpm_display = np.zeros((bpm_height, bpm_width, 3), np.uint8)
    ### 
    bpm_text_size, bpm_text_base = cv2.getTextSize(ear_str, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2.7,
                                                   thickness=2)
    bpm_text_x = int((bpm_width - bpm_text_size[0]) / 2)
    bpm_text_y = int(bpm_height * 0.2 + bpm_text_base)
    cv2.putText(bpm_display,ear_str , (bpm_text_x, bpm_text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1.7, color=(0, 0, 255), thickness=2)
    bpm_label_size, bpm_label_base = cv2.getTextSize('EAR', fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                                     thickness=1)
    bpm_label_x  = int(bpm_width * 0.7) #int((bpm_width - bpm_label_size[0]) / 2)
    #bpm_label_y = int(bpm_height - bpm_label_size[1] * 2)
    bpm_label_y = int(bpm_height * 0.2 + bpm_text_base)
    cv2.putText(bpm_display, 'ear', (bpm_label_x, bpm_label_y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.5, color=(0, 0, 255), thickness=1)
    
    ### 
    bpm_text_size, bpm_text_base = cv2.getTextSize(bpm_str, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2.7,
                                                   thickness=2)
    bpm_text_x = int((bpm_width - bpm_text_size[0]) / 2)
    bpm_text_y = int(bpm_height *0.7 + bpm_text_base)
    cv2.putText(bpm_display,bpm_str, (bpm_text_x, bpm_text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1.7, color=(0, 255, 0), thickness=2)
    bpm_label_size, bpm_label_base = cv2.getTextSize('BPM', fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                                     thickness=1)
    bpm_label_x  = int(bpm_width * 0.7) #
    bpm_label_y = int(bpm_height * 0.7 + bpm_text_base)
    cv2.putText(bpm_display, 'BPM', (bpm_label_x, bpm_label_y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=(0, 255, 0), thickness=1)
       
    return bpm_display


def draw_fps(frame, fps):
    cv2.rectangle(frame, (0, 0), (100, 30), color=(0, 0, 0), thickness=1)
    cv2.putText(frame, 'FPS: ' + str(round(fps, 2)), (5, 20), fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1, color=(0, 255, 0))
    return frame


def draw_graph_text(text, color, graph_width, graph_height):
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    text_size, text_base = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=1)
    text_x = int((graph_width - text_size[0]) / 2)
    text_y = int((graph_height / 2 + text_base))
    cv2.putText(graph, text, (text_x, text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=color,
                thickness=1)
    return graph


def compute_bpm(filtered_values, fps, buffer_size, last_bpm):
    fft = np.abs(np.fft.rfft(filtered_values))

    freqs = fps / buffer_size * np.arange(buffer_size / 2 + 1)

    while True:
        max_idx = fft.argmax()
        bps = freqs[max_idx]
        if bps < MIN_HZ or bps > MAX_HZ:
            if DEBUG_MODE:
                print( 'BPM of {0} was discarded.'.format(bps * 60.0))
            fft[max_idx] = 0
        else:
            bpm = bps * 60.0
            break

    if last_bpm > 0:
        bpm = (last_bpm * 0.9) + (bpm * 0.1)

    return bpm


def filter_signal_data(values, fps):
    values = np.array(values)
    np.nan_to_num(values, copy=False)

    detrended = signal.detrend(values, type='linear')
    demeaned = sliding_window_demean(detrended, 15)
    filtered = butterworth_filter(demeaned, MIN_HZ, MAX_HZ, fps, order=5)
    return filtered


def get_roi_avg(frame, view, face_points, draw_rect=True):
    fh_left, fh_right, fh_top, fh_bottom = get_forehead_roi(face_points)
    nose_left, nose_right, nose_top, nose_bottom = get_nose_roi(face_points)

    if draw_rect:
        cv2.rectangle(view, (fh_left, fh_top), (fh_right, fh_bottom), color=(0, 255, 0), thickness=1)
        cv2.rectangle(view, (nose_left, nose_top), (nose_right, nose_bottom), color=(0, 255, 0), thickness=1)

    # Slice out the regions of interest (ROI) and average them
    fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
    nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]
    return get_avg(fh_roi, nose_roi)


def run_pulse_observer(detector, predictor, webcam, window):
    roi_avg_values = []
    graph_values = []
    graph_eye_values = []
    times = []
    last_bpm = 0
    graph_height = 160   #480
    graph_width = 340 #int(200 * 0.75)
    bpm_display_width = 340

    while cv2.getWindowProperty(window, 0) == 0:
        ret_val, frame = webcam.read()

        if not ret_val:
            print(  "ERROR:  Unable to read from webcam.  Was the webcam disconnected?  Exiting.")
            shut_down(webcam)

        view = np.array(frame)

        if graph_width == 0:
            graph_width = int(view.shape[1] * 0.75)
            if DEBUG_MODE:
                print(  'Graph width = {0}'.format(graph_width) )
        if bpm_display_width == 0:
            bpm_display_width = view.shape[1] - graph_width

        faces = detector(frame, 0)
        if len(faces) == 1:
            face_points = predictor(frame, faces[0])
            #---------------------- face ......
            shape = face_utils.shape_to_np( face_points ) #shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)            
            
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(view, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(view, [rightEyeHull], -1, (0, 0, 255), 1) 
            
            ear = (leftEAR + rightEAR) / 2.0            
                        
            #print( ear )
            graph_eye_values.append( ear  )
            if len(graph_eye_values) > MAX_VALUES_TO_GRAPH:
                                graph_eye_values.pop(0)
                                        
            #---------------------- face ......                       
            roi_avg = get_roi_avg(frame, view, face_points, draw_rect=True)
            roi_avg_values.append(roi_avg)
            times.append(time.time())

            if len(times) > BUFFER_MAX_SIZE:
                roi_avg_values.pop(0)
                times.pop(0)

            curr_buffer_size = len(times)

            if curr_buffer_size > MIN_FRAMES:                
                time_elapsed = times[-1] - times[0]
                fps = curr_buffer_size / time_elapsed   
                filtered = filter_signal_data(roi_avg_values, fps)

                graph_values.append(filtered[-1])
                if len(graph_values) > MAX_VALUES_TO_GRAPH:
                    graph_values.pop(0)
                
                graph  = draw_graph(graph_values, graph_width, graph_height )
                graph2 = draw_graph2(graph_eye_values, graph_width, graph_height )
                bpm = compute_bpm(filtered, fps, curr_buffer_size, last_bpm)
                
                ss = '{0:0.3f}'.format(ear)
                bpm_display = draw_bpm(ss,str(int(round(bpm))), bpm_display_width, graph_height)
                last_bpm = bpm
                if DEBUG_MODE:
                    view = draw_fps(view, fps)

            else:
                pct = int(round(float(curr_buffer_size) / MIN_FRAMES * 100.0))
                loading_text = 'Computing: ' + str(pct) + '%'
                graph  = draw_graph_text(loading_text, (0, 255, 0), graph_width, graph_height )
                graph2 = draw_graph2(graph_eye_values, graph_width, graph_height )
                
                ss = '{0:0.3f}'.format(ear)
                bpm_display = draw_bpm(ss,"wait", bpm_display_width, graph_height)
                last_bpm = 0 
                
 
        else:
            del roi_avg_values[:]
            del times[:]
            graph  = draw_graph_text('No face detected', (0, 255, 0), graph_width, graph_height )
            graph2 = draw_graph_text('No face detected', (0, 0, 255), graph_width, graph_height )
            bpm_display = draw_bpm('--', '--', bpm_display_width, graph_height )

 #       graph = np.hstack((graph, bpm_display))
 #       view = np.vstack((view, graph))
        graph2  = np.vstack((graph2, bpm_display))
        graph2 = np.vstack((graph2, graph))
        view  = np.hstack((view, graph2))

        cv2.imshow(window, view)

        key = cv2.waitKey(1)
        if key == 27:
            shut_down(webcam)

def shut_down(webcam):
    webcam.release()
    cv2.destroyAllWindows()
    exit(0)


def main():
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor('d://68_face_landmarks.dat')
    except RuntimeError as e:
        return
    
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print(  'ERROR:  Unable to open webcam.  Verify that webcam is connected and try again.  Exiting.')
        webcam.release()
        return

    cv2.namedWindow(WINDOW_TITLE)
    run_pulse_observer(detector, predictor, webcam, WINDOW_TITLE)

    shut_down(webcam)


if __name__ == '__main__':
    main()