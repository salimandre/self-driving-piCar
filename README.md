# Self driving piCar

## Lanes detection

we use the following preprocessing pipeline:
* Load in **BGR** color space
* **HSV** color space transform
* Create a binary mask by **thresholding**
* **Blur** mask
* Apply **Canny filter**
* **Crop** top half image 
* Apply **Probabilistic Hough Transform**
* Compute **centroids** and mean direction on patches
* Compute **weighted polynomial interpolation** over centroids
* Compute **steering angle**

<img src="img/pipeline_angle.gif" width="40%">
