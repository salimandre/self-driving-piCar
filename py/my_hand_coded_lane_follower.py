import cv2
import numpy as np
import logging
import math
import datetime
import sys

_SHOW_IMAGE = False


class MyHandCodedLaneFollower(object):

    def __init__(self, car=None):
        logging.info('Creating a HandCodedLaneFollower...')
        self.car = car
        self.curr_steering_angle = 90

    def follow_lane(self, frame):
        # Main entry point of the lane follower
        #show_image("orig", frame)

        frame_lanes, new_steering_angle = pipeline_lane_detector(frame, self.curr_steering_angle)
        self.curr_steering_angle = new_steering_angle

        #cv2.imshow('lanes tracking', frame_lanes)

        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
            #self.car.back_wheels.speed = 0

        return frame_lanes

############################
# Frame processing steps
############################
def draw_patches(img,patch_):
    cv2.line(img,(patch_['x'][0],patch_['y'][0]),(patch_['x'][1],patch_['y'][0]),(0,165,255),1)
    cv2.line(img,(patch_['x'][0],patch_['y'][1]),(patch_['x'][1],patch_['y'][1]),(0,165,255),1)
    cv2.line(img,(patch_['x'][0],patch_['y'][0]),(patch_['x'][0],patch_['y'][1]),(0,165,255),1)
    cv2.line(img,(patch_['x'][1],patch_['y'][0]),(patch_['x'][1],patch_['y'][1]),(0,165,255),1)
    return img

def get_centroids_from_patches(list_lines,patch_):
    centroids = {'bottom': np.zeros((1,2)),'top': np.zeros((1,2))}
    velocity = (None,None)

    for line in list_lines:
        x1,y1,x2,y2 = line[0]
        if x1>=patch_['x'][0] and x1<=patch_['x'][1] and y1<=patch_['y'][0] and y1>=patch_['y'][1]:
            if x2>=patch_['x'][0] and x2<=patch_['x'][1] and y2<=patch_['y'][0] and y2>=patch_['y'][1]:
                centroids['bottom'] = np.vstack((centroids['bottom'],np.array([x1,y1])))
                centroids['top'] = np.vstack((centroids['top'],np.array([x2,y2])))  

    centroids['bottom'] = centroids['bottom'][1:,:]
    centroids['top'] = centroids['top'][1:,:]

    if len(centroids['bottom'])>0:
        for arrow_side in ['bottom','top']:
                centroids[arrow_side] = np.mean(centroids[arrow_side],axis=0)

        if centroids['bottom'][1] <= centroids['top'][1]:
            velocity = (centroids['bottom'][0]-centroids['top'][0], centroids['bottom'][1]-centroids['top'][1])
        else:
            velocity = (centroids['top'][0]-centroids['bottom'][0], centroids['top'][1]-centroids['bottom'][1])

        for arrow_side in ['bottom','top']:
            centroids[arrow_side] = [int(np.round(a)) for a in centroids[arrow_side]]

        empty_bool = False

    else:
        empty_bool = True

    return centroids, velocity, empty_bool

def stabilize_steering_angle(curr_steering_angle, last_steering_angle=None, alpha=0.2):
    if last_steering_angle is None:
        return int(curr_steering_angle)
    else:
        if 135-last_steering_angle<=5 and curr_steering_angle>= last_steering_angle:
            return np.clip(int(alpha * curr_steering_angle + (1.-alpha) * last_steering_angle),last_steering_angle-1,last_steering_angle+1)
        elif last_steering_angle-55<=5 and curr_steering_angle<= last_steering_angle:
            return np.clip(int(alpha * curr_steering_angle + (1.-alpha) * last_steering_angle),last_steering_angle-1,last_steering_angle+1)
        else:
            return np.clip(int(alpha * curr_steering_angle + (1.-alpha) * last_steering_angle),last_steering_angle-3,last_steering_angle+3)

def pipeline_lane_detector(frame_, past_steering_angle=None):

    row_threshold = 120

    img_rgb = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
    img_top_half_bgr = frame_[:-row_threshold,:]

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    img_bottom_half_bgr = cv2.cvtColor(img_rgb[-row_threshold:,:], cv2.COLOR_RGB2BGR)
    img_crop_hsv = img_hsv[-row_threshold:,:]

    bound = (np.array([0, 0, 0]), np.array([255, 255, 50]))

    mask = cv2.inRange(img_crop_hsv, bound[0], bound[1])

    #mask_blurred = cv2.blur(mask,(5,5))

    mask_edges = cv2.Canny(mask, 200, 400)

    minLineLength = 12
    maxLineGap = 3
    min_threshold = 5
    lines = cv2.HoughLinesP(mask_edges,1,np.pi/180,min_threshold,minLineLength,maxLineGap)
    """
    list_patch = [{'x': (0,25),'y': (100,75)}, {'x': (25,50),'y': (100,75)}, {'x': (50,75),'y': (100,75)},{'x': (75,100),'y': (100,75)}, {'x': (100,125),'y': (100,75)}]
    list_patch += [{'x': (194,219),'y': (100,75)},{'x': (219,244),'y': (100,75)}, {'x': (244,269),'y': (100,75)}, {'x': (269,294),'y': (100,75)},{'x': (294,319),'y': (100,75)}]

    list_patch += [{'x': (0,25),'y': (75,50)}, {'x': (25,50),'y': (75,50)}, {'x': (50,75),'y': (75,50)},{'x': (75,100),'y': (75,50)}, {'x': (100,125),'y': (75,50)}]
    list_patch += [{'x': (194,219),'y': (75,50)},{'x': (219,244),'y': (75,50)}, {'x': (244,269),'y': (75,50)}, {'x': (269,294),'y': (75,50)},{'x': (294,319),'y': (75,50)}]

    list_patch += [{'x': (0,25),'y': (50,25)}, {'x': (25,50),'y': (50,25)}, {'x': (50,75),'y': (50,25)},{'x': (75,100),'y': (50,25)}, {'x': (100,125),'y': (50,25)}]
    list_patch += [{'x': (194,219),'y': (50,25)},{'x': (219,244),'y': (50,25)}, {'x': (244,269),'y': (50,25)}, {'x': (269,294),'y': (50,25)},{'x': (294,319),'y': (50,25)}]

    list_patch += [{'x': (0,25),'y': (25,0)}, {'x': (25,50),'y': (25,0)}, {'x': (50,75),'y': (25,0)},{'x': (75,100),'y': (25,0)}, {'x': (100,125),'y': (25,0)}]
    list_patch += [{'x': (194,219),'y': (25,0)},{'x': (219,244),'y': (25,0)}, {'x': (244,269),'y': (25,0)}, {'x': (269,294),'y': (25,0)},{'x': (294,319),'y': (25,0)}]
    """
    list_patch = [{'x': (0,25),'y': (120,100)}, {'x': (25,50),'y': (120,100)}, {'x': (50,75),'y': (120,100)},{'x': (75,100),'y': (120,100)}, {'x': (100,125),'y': (120,100)}]
    list_patch += [{'x': (194,219),'y': (120,100)},{'x': (219,244),'y': (120,100)}, {'x': (244,269),'y': (120,100)}, {'x': (269,294),'y': (120,100)},{'x': (294,319),'y': (120,100)}]

    list_patch += [{'x': (0,25),'y': (100,75)}, {'x': (25,50),'y': (100,75)}, {'x': (50,75),'y': (100,75)},{'x': (75,100),'y': (100,75)}, {'x': (100,125),'y': (100,75)}]
    list_patch += [{'x': (194,219),'y': (100,75)},{'x': (219,244),'y': (100,75)}, {'x': (244,269),'y': (100,75)}, {'x': (269,294),'y': (100,75)},{'x': (294,319),'y': (100,75)}]

    list_patch += [{'x': (0,25),'y': (75,50)}, {'x': (25,50),'y': (75,50)}, {'x': (50,75),'y': (75,50)},{'x': (75,100),'y': (75,50)}, {'x': (100,125),'y': (75,50)}]
    list_patch += [{'x': (194,219),'y': (75,50)},{'x': (219,244),'y': (75,50)}, {'x': (244,269),'y': (75,50)}, {'x': (269,294),'y': (75,50)},{'x': (294,319),'y': (75,50)}]

    list_patch += [{'x': (75,100),'y': (50,25)}, {'x': (100,125),'y': (50,25)}]
    list_patch += [{'x': (194,219),'y': (50,25)},{'x': (219,244),'y': (50,25)}]

    list_patch += [{'x': (100,125),'y': (25,0)}]
    list_patch += [{'x': (194,219),'y': (25,0)}]

    for patch in list_patch:
        img_bottom_half_bgr = draw_patches(img_bottom_half_bgr, patch)

    if lines is None:
        return frame_, past_steering_angle
    else:
        X_left=np.zeros((1,2))
        X_right=np.zeros((1,2))
        n_right_side_right_dir = 0
        n_right_side_left_dir = 0
        n_left_side_right_dir = 0
        n_left_side_left_dir = 0        
        for patch in list_patch:
            centroids, velocity, empty_bool = get_centroids_from_patches(lines, patch)
            if not empty_bool:
                if velocity[1] < -0.25 and centroids['bottom'][0]>160: #velocity[0] < -0.25 and 
                    X_right = np.vstack((X_right, centroids['bottom']))
                    n_right_side_left_dir += int(velocity[0] < -0.25)
                    n_right_side_right_dir += int(velocity[0] >= -0.25)
                elif velocity[1] < -0.25 and centroids['bottom'][0]<160: #velocity[0] > 0.25 and 
                    X_left = np.vstack((X_left, centroids['bottom']))
                    n_left_side_left_dir += int(velocity[0] < -0.25)
                    n_left_side_right_dir += int(velocity[0] >= -0.25)

        if n_right_side_right_dir>=n_right_side_left_dir:
            right_side_dir = ('right', n_right_side_right_dir)
        else:
            right_side_dir = ('left', n_right_side_left_dir)

        if n_left_side_right_dir>=n_left_side_left_dir:
            left_side_dir = ('right', n_left_side_right_dir)
        else:
            left_side_dir = ('left', n_left_side_left_dir)

        if right_side_dir[0] == 'right' and left_side_dir[0] == 'right':
            X_right=np.zeros((1,2))

        if right_side_dir[0] == 'left' and left_side_dir[0] == 'left':
            X_left=np.zeros((1,2))

        X_left = X_left[1:,:]
        X_right = X_right[1:,:]

        if len(X_right)>1:
            right_lane = np.polyfit(X_right[:,0],X_right[:,1],1,w=X_right[:,1])
            y1 = right_lane[0] * 219 + right_lane[1]
            y2 = right_lane[0] * 319 + right_lane[1]
            cv2.line(img_bottom_half_bgr,(219,int(y1)),(319,int(y2)),(150,50,240),5) #150,50,255

            x_start_right = int((25 - right_lane[1])/(right_lane[0]+0.001))
        if len(X_left)>1:

            left_lane = np.polyfit(X_left[:,0],X_left[:,1],1,w=X_left[:,1])
            y1 = left_lane[0] * 0 + left_lane[1]
            y2 = left_lane[0] * 100 + left_lane[1]
            cv2.line(img_bottom_half_bgr,(0,int(y1)),(100,int(y2)),(150,50,240),5)

            x_start_left = int((25 - left_lane[1])/(left_lane[0]+0.001))

        if len(X_right)>1 and len(X_left)>1:
            mid_star = 0.5 * (x_start_right + x_start_left)
            cv2.line(img_bottom_half_bgr,(int(np.clip(mid_star,-10000,10000)),25),(160,100),(0,0,255),5)
        elif len(X_right)>1 and len(X_left)==0:
            #mid_star = 25/right_lane[0] - 160 - right_lane[1]/right_lane[0]
            mid_star = (25-100)/right_lane[0] + 160
            cv2.line(img_bottom_half_bgr,(int(np.clip(mid_star,-10000,10000)),25),(160,100),(0,0,255),5)
        elif len(X_right)==0 and len(X_left)>1:
            mid_star = (25-100)/left_lane[0] + 160
            cv2.line(img_bottom_half_bgr,(int(np.clip(mid_star,-10000,10000)),25),(160,100),(0,0,255),5)
        else:
            mid_star = 159

        if np.abs(mid_star-160)<2:
            steering_angle = 90
        else:
            steering_angle = 90 + np.degrees(np.arctan((mid_star-160)/75.))
            steering_angle = np.clip(steering_angle,55, 135)

        stable_steering_angle = stabilize_steering_angle(steering_angle,past_steering_angle)

        # setup text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(stable_steering_angle)
        # get boundary of this text
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        # get coords based on boundary
        textX = 110
        textY = 30
        # add text centered on image
        cv2.putText(img_bottom_half_bgr, text, (textX, textY ), font, 1, (0, 0, 255), 2)

        new_frame = np.concatenate((img_top_half_bgr, img_bottom_half_bgr), axis=0)
        return new_frame, stable_steering_angle


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    test_video('/home/pi/DeepPiCar/driver/data/tmp/video01')
    #test_photo('/home/pi/DeepPiCar/driver/data/video/car_video_190427_110320_073.png')
    #test_photo(sys.argv[1])
    #test_video(sys.argv[1])
