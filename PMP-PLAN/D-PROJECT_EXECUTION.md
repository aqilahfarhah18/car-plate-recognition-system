# PROJECT OVERVIEW
## D. EXECUTING THE PROJECT
### Project Design and Coding
Flowchart Design:
<img src="assets/flowchart.JPG" width="60%">

 Car Plate Implementation Example

### Description of the project coding and implementation
Libraries and Packages required:


Car Plate Detection: 
Firstly, we load our model from the pre-trained which created by sergiomsilva.From our Github repository, two files: wpod-net.h5 and wpod-net.json canbe found. Notifications "loading model successfully..." should be get after this step. 
<img src="assets/requi.JPG" width="40%">

Subsequently, we create a function name preprocess_image to read and pre-process our plate images. This function basically reads the parsing image, converts it to RGB (line 3) and normalizes the image data to 0–1 range so it can be compatible with matplotlib. Additionally, we can set resize = True to resize all images to same dimension of (width = 224, height = 224) for visualizing purpose in the next step.
<img src="assets/1.JPG" width="100%">

In this step, we visualize our vehicle dataset. This dataset contains of 20 vehicle images with plate acquired from 10 different countries (Germany, Vietnam, Japan, Thailand, Saudi, Russia, Korea, Usa, India, China). The following block of code will display plate images and their country names in a figure containing of 5 columns and 4 rows.
<img src="assets/2.JPG" width="100%">

Now, function named get_plate which processes the raw image is written, and is sent to our model and return the plate image (LpImg) and its coordinates (cor). If there is no plate founded, the program would warn with an error “No License plate is founded!”. 
<img src="assets/3.JPG" width="100%">

We draw a bounding box with obtained coordinates of deteced plate. 
<img src="assets/4.JPG" width="100%">

We perform get_plate function to all vehicle images and plot the returend plate images. 
<img src="assets/5.JPG" width="100%">

Plate Character Segmentation with OpenCV:
The image undergoes image processing by converting to grayscale, blur image, image thresholding and dilation. 
<img src="assets/6.JPG" width="100%">

 findContours function of OpenCV is used to identify the coordinates of license character. This function is based on a simple theory: contours is simply curve joining all continuous point (along the boundary) sharing the same color and intensity. We created a function called sort_contours which basically sorts founded contours from left to right. We add another contour filter. Later, we draw bounding with all contours that passes these filters, applies binary thresholding on each determined contours and append them to list crop_characters.
<img src="assets/7.JPG" width="100%">

 Ater having all segmented characters storing in crop_characters. We visualize the with matplotlib using the example code as below. We train Neural Network model which is able to recognize and convert those characters to digital letters.
<img src="assets/8.JPG" width="100%">

### Project Result
Result using Tkinter:
<img src="assets/tkinter.JPG" width="100%">

Result using Web API:
<img src="assets/9.jpeg" width="100%">


<br><br><br>
##### Next: [Project Closing](E-PROJECT_CLOSING.md)

