# self-driving-piCar

## Lanes detection

we use the following preprocessing pipeline:
* Load in **BGR** color space
* **HSV** color space transform
* Create a binary mask by **thresholding**
* **Blur** mask
* Apply **Canny filter**
* **Crop** top half image 
* Apply **Probabilistic Hough Transform**

<img src="img/pipeline_lane.gif" width="40%">
