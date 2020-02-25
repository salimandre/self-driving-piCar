# Self driving piCar

## Lanes detection

we use the following preprocessing pipeline:
* Load in **RGB** color space
* Perform **HSV** color space transform
* Create a binary mask by **thresholding**
* Apply **Gaussian blur** on mask
* Apply **Canny filter**
* **Crop** top half image 
* Apply **Probabilistic Hough Transform**
* Compute **centroids** and mean direction on patches
* Compute **weighted polynomial interpolation** over centroids
* Compute **steering angle**

<img src="img/pipeline_angle.gif" width="40%">

## Results

Few **comments**:

- We removed the **blur** since with the speed there is already blur and actually the **magnitude parameter** when using canny filter needs to be tuned accordingly.

- Since the steering angle computed is noisy we smooth it by applying the following **update**:

theta_new <- alpha * theta_new + (1-alpha) * theta_old

- The camera provided with the car has **only 120 degree angle** view, which is a bit short. In the reference https://github.com/dctian/DeepPiCar we followed they use a 170 degree angle camera and noticed significant improvements.

- **Black lanes** are very **sensitive to illumination** and choosing colored lanes would probably improve results.

- Our front wheels are not yet properly **calibrated** and even though steering angle is smoothly updated the wheels tend to have too much angle when turning left side which slows down the car  

### Car view

**Note**: The steering angle indicated by text is **smoothed** while the steering angle represented by the red lane is the **raw** current steering angle.

<img src="img/demo_car_view_cleaned.gif" width="40%">

### Outside view

<img src="img/final_demo_outside.gif" width="40%">
