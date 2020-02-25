import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt 

IMG_PATH = '/Users/mac/Desktop/python/pivision/img/'

"""
img_bgr = cv2.imread(IMG_PATH+'2020-02-16-105817.jpg')
#img_grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

#cv2.imshow('grey image',img_grey)
#cv2.waitKey(2000)
cv2.imshow('BGR image',img_bgr)
cv2.waitKey(3000)
cv2.imshow('HSV image',img_hsv)
cv2.waitKey(3000)

bound = (np.array([0, 0, 0]), np.array([255, 255, 50]))

mask = cv2.inRange(img_hsv, bound[0], bound[1])

cv2.imshow('mask',mask)
cv2.waitKey(0)
"""

def draw_patches(img,patch_):
	cv2.line(img,(patch_['x'][0],patch_['y'][0]),(patch_['x'][1],patch_['y'][0]),(255,165,0),1)
	cv2.line(img,(patch_['x'][0],patch_['y'][1]),(patch_['x'][1],patch_['y'][1]),(255,165,0),1)
	cv2.line(img,(patch_['x'][0],patch_['y'][0]),(patch_['x'][0],patch_['y'][1]),(255,165,0),1)
	cv2.line(img,(patch_['x'][1],patch_['y'][0]),(patch_['x'][1],patch_['y'][1]),(255,165,0),1)
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

def pipeline_lane_detector(path_to_img, sleep_time=1):

	row_threshold = 210

	img_bgr = cv2.imread(path_to_img)
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

	img_rgb = img_rgb[-2*row_threshold:,:]

	plt.imshow(img_rgb)
	plt.title('original RGB image')
	plt.xticks([])
	plt.yticks([])
	plt.draw()
	plt.pause(sleep_time)

	img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

	plt.imshow(img_hsv)
	plt.title('HSV image')
	plt.draw()
	plt.pause(sleep_time)

	img_crop_rgb = img_rgb[len(img_rgb)-row_threshold:,:]
	img_crop_hsv = img_hsv[row_threshold:,:]

	plt.imshow(np.vstack((img_crop_rgb,img_crop_hsv)))
	plt.title('cropped image')
	plt.draw()
	plt.pause(sleep_time)

	bound = (np.array([0, 0, 0]), np.array([255, 255, 50]))

	mask = cv2.inRange(img_crop_hsv, bound[0], bound[1])

	plt.imshow(np.vstack((img_crop_rgb,np.transpose(np.broadcast_to(mask,(3,mask.shape[0],mask.shape[1])), (1, 2, 0)) )))
	plt.title('masking')
	plt.draw()
	plt.pause(sleep_time)

	mask_blurred = cv2.blur(mask,(5,5))

	plt.imshow(np.vstack((img_crop_rgb,np.transpose(np.broadcast_to(mask_blurred,(3,mask.shape[0],mask.shape[1])), (1, 2, 0)) )))
	plt.title('Gaussian blurred mask')
	plt.draw()
	plt.pause(sleep_time)

	mask_edges = cv2.Canny(mask_blurred, 200, 400)

	plt.imshow(np.vstack((img_crop_rgb,np.transpose(np.broadcast_to(mask_edges,(3,mask.shape[0],mask.shape[1])), (1, 2, 0)) )))
	plt.title('Canny filter')
	plt.draw()
	plt.pause(sleep_time)

	minLineLength = 12
	maxLineGap = 3
	min_threshold = 5
	lines = cv2.HoughLinesP(mask_edges,1,np.pi/180,min_threshold,minLineLength,maxLineGap)

	hough_mask = cv2.cvtColor(np.zeros(mask_edges.shape, np.uint8), cv2.COLOR_GRAY2BGR)
	for line in lines:
		x1,y1,x2,y2 = line[0]
		cv2.line(hough_mask,(x1,y1),(x2,y2),(0,255,0),2)
	
	plt.imshow(np.vstack((img_crop_rgb,hough_mask)))
	plt.title('Hough transform')
	plt.draw()
	plt.pause(sleep_time)

	list_patch = [{'x': (0,50),'y': (200,150)}, {'x': (50,100),'y': (200,150)}, {'x': (100,150),'y': (200,150)},{'x': (150,200),'y': (200,150)}]
	list_patch += [{'x': (439,489),'y': (200,150)}, {'x': (489,539),'y': (200,150)}, {'x': (539,589),'y': (200,150)},{'x': (589,639),'y': (200,150)}]

	list_patch += [{'x': (0,50),'y': (150,100)}, {'x': (50,100),'y': (150,100)}, {'x': (100,150),'y': (150,100)},{'x': (150,200),'y': (150,100)}]
	list_patch += [{'x': (439,489),'y': (150,100)}, {'x': (489,539),'y': (150,100)}, {'x': (539,589),'y': (150,100)},{'x': (589,639),'y': (150,100)}]

	list_patch += [{'x': (0,50),'y': (100,50)}, {'x': (50,100),'y': (100,50)}, {'x': (100,150),'y': (100,50)},{'x': (150,200),'y': (100,50)}]
	list_patch += [{'x': (439,489),'y': (100,50)}, {'x': (489,539),'y': (100,50)}, {'x': (539,589),'y': (100,50)},{'x': (589,639),'y': (100,50)}]

	list_patch += [{'x': (0,50),'y': (50,0)}, {'x': (50,100),'y': (50,0)}, {'x': (100,150),'y': (50,0)},{'x': (150,200),'y': (50,0)}]
	list_patch += [{'x': (439,489),'y': (50,0)}, {'x': (489,539),'y': (50,0)}, {'x': (539,589),'y': (50,0)},{'x': (589,639),'y': (50,0)}]
	
	X_left=np.zeros((1,2))
	X_right=np.zeros((1,2))
	for patch in list_patch:
		hough_mask = draw_patches(hough_mask, patch)

	plt.imshow(np.vstack((img_crop_rgb,hough_mask)))
	plt.title('patches')
	plt.draw()
	plt.pause(sleep_time)

	for patch in list_patch:
		centroids, velocity, empty_bool = get_centroids_from_patches(lines, patch)
		if not empty_bool:
			cv2.circle(img=hough_mask, center=tuple(centroids['bottom']), radius=1, color=(255,255,0), thickness=6)#, lineType=8, shift=0)	(255,191,0)
			if velocity[0] < -0.25 and velocity[1] < -0.25 and centroids['bottom'][0]>300:
				X_right = np.vstack((X_right, centroids['bottom']))
			elif velocity[0] > 0.25 and velocity[1] < -0.25 and centroids['bottom'][0]<300:
				X_left = np.vstack((X_left, centroids['bottom']))

	plt.imshow(np.vstack((img_crop_rgb,hough_mask)))
	plt.title('centroids')
	plt.draw()
	plt.pause(sleep_time)

	X_left = X_left[1:,:]
	X_right = X_right[1:,:]

	if len(X_right)>1:
		right_lane = np.polyfit(X_right[:,0],X_right[:,1],1,w=X_right[:,1])
		y1 = right_lane[0] * 439 + right_lane[1]
		y2 = right_lane[0] * 639 + right_lane[1]
		cv2.line(hough_mask,(439,int(y1)),(639,int(y2)),(255,50,150),2) #150,50,255

		x_start_right = int((50 - right_lane[1])/right_lane[0])
	if len(X_left)>1:

		left_lane = np.polyfit(X_left[:,0],X_left[:,1],1,w=X_left[:,1])
		y1 = left_lane[0] * 0 + left_lane[1]
		y2 = left_lane[0] * 200 + left_lane[1]
		cv2.line(hough_mask,(0,int(y1)),(200,int(y2)),(255,50,150),2)

		x_start_left = int((50 - left_lane[1])/left_lane[0])

	#print('x star right: ', x_start_right)
	#print('x star left: ', x_start_left)

	plt.imshow(np.vstack((img_crop_rgb,hough_mask)))
	plt.title('polynomial interpolation')
	plt.draw()
	plt.pause(sleep_time)

	if len(X_right)>1 and len(X_left)>1:
		mid_star = 0.5 * (x_start_right + x_start_left)
		cv2.line(hough_mask,(int(mid_star),50),(320,200),(255,0,0),4)
	elif len(X_right)>1 and len(X_left)==0:
		mid_star = 50/right_lane[0] - 320 - right_lane[1]/right_lane[0]
		cv2.line(hough_mask,(int(mid_star),50),(320,200),(255,0,0),4)
	elif len(X_right)==0 and len(X_left)>1:
		mid_star = 50/left_lane[0] - 320 - left_lane[1]/left_lane[0]
		cv2.line(hough_mask,(int(mid_star),50),(320,200),(255,0,0),4)

	plt.imshow(np.vstack((img_crop_rgb,hough_mask)))
	plt.title('steering angle')
	plt.draw()
	plt.pause(sleep_time)

	steering_angle = np.arctan(150./(mid_star-320))
	steering_angle = int(np.clip(90+steering_angle ,45, 135))

	# setup text
	font = cv2.FONT_HERSHEY_SIMPLEX
	text = str(steering_angle)
	# get boundary of this text
	textsize = cv2.getTextSize(text, font, 1, 2)[0]
	# get coords based on boundary
	textX = 275
	textY = 250
	# add text centered on image
	angle_img = np.vstack((img_crop_rgb,hough_mask))
	cv2.putText(angle_img, text, (textX, textY ), font, 1, (0, 0, 255), 2)
	cv2.imshow('tadaaaa', angle_img)
	cv2.waitKey(0)

	plt.savefig('final_preprocessing.png')


# ------------------------------------------------------------------------------------------------------------------------------


"""
for file in os.listdir(IMG_PATH):
	if file.endswith(".jpg"):
		print(file)
		pipeline_lane_detector(IMG_PATH+file, 1)

"""

pipeline_lane_detector(IMG_PATH+'2020-02-16-105817.jpg',2) #004507 #105817 #110814


