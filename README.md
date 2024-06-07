# Deep-Learning-Project-Fruit-Image-Classification
	

<h3>
  
VGG 16 =>
Train : 97%
Test : 70%

AlexNet	=>
Train:98%
Test : 69%

CNN =>
Train:95%
Test : 65%

GoogleNet =>
Train:94%
Test : 64%

LeNet =>
Train:98%
Test : 71%

Transformer =>
Train:98%
Test : 75%
</h3>



<pre>
<h3>Preprocessing Step:</h3>
-	Image Resizing.
-	Encoding for Labels.
</pre>
<pre>
<h3>Augmentation Process:</h3>
-	rescale=1. / 255
-	shear_range=0.2
-	zoom_range=0.2
-	horizontal_flip=True
-	rotation_range=10
-	width_shift_range=0.1
-	height_shift_range=0.1
-	fill_mode='nearest'
-	brightness_range=[0.5, 1.5],  # Adjust brightness
-	channel_shift_range=50.0  # Adjust channel intensity
-	vertical_flip=True  # Flip vertically
-	featurewise_center=False
-	featurewise_std_normalization=False
-	zca_whitening=False
</pre>
<h3>can load and save the model as model.tfl</h3>


<h3>The script processes each image in a specified directory, predicts its label using a pre-trained CNN model, and writes the results to a CSV file named final.csv. The results include the image ID (without the file extension) and the index of the predicted label (starting from 1). The script uses OpenCV for image preprocessing, NumPy for handling arrays, and the CSV module for writing results to a file.
</h3>
