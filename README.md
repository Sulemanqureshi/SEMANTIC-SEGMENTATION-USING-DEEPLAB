# SEMANTIC-SEGMENTATION-USING-DEEPLAB
Imports:
torch: Imports PyTorch, a deep learning library.
torchvision: Imports TorchVision, which provides pre-trained models and image utilities.
cv2: Imports OpenCV for reading, manipulating, and processing images.
numpy: Imports NumPy for numerical operations such as array manipulation.
matplotlib.pyplot: Imports Matplotlib for visualizing images.
PIL.Image: Imports Python Imaging Library (PIL) for image handling.
Load DeepLabV3 Model:
Load pre-trained DeepLabV3 model: Loads the DeepLabV3 model with a ResNet-101 backbone that is pre-trained for semantic segmentation tasks (deeplabv3_resnet101).
Set model to evaluation mode: Switches the model to evaluation mode using model.eval(), which disables training-specific behaviors like dropout.
Preprocessing Function (preprocess):
Load the image: Reads the image using cv2.imread(), which loads it in BGR format (the default for OpenCV).
Convert BGR to RGB: Converts the image from BGR to RGB using cv2.cvtColor() for compatibility with the PyTorch model.
Resize the image: Resizes the image to (512, 512) pixels to match the input size required by the model using cv2.resize().
Create preprocessing transformation: Uses torchvision.transforms.Compose() to define a preprocessing pipeline that converts the image into a tensor and normalizes it using ImageNet’s mean and standard deviation values.
Convert the image to a tensor: Converts the resized image into a tensor and adds a batch dimension using .unsqueeze(0).
Return the preprocessed tensor and resized image: The function returns the input tensor (for inference) and the resized image (for visualization).
Segmentation Function (segment_image):
Disable gradient computation: Uses torch.no_grad() to disable gradient calculations, making inference faster.
Run the model: Passes the input tensor through the DeepLabV3 model to get segmentation results, which are stored in the 'out' key.
Extract class predictions: Takes the argmax of the model's output across the class dimension to get the class prediction for each pixel. The result is converted to a NumPy array using .cpu().numpy().
Return the predicted segmentation mask: The function returns the segmentation mask, where each pixel is labeled with the class index predicted by the model.
Mask Overlay Function (overlay_segmentation_mask):
Create a color map for classes: Generates a random colormap with 21 colors (the number of classes in the DeepLabV3 model), where each class is mapped to a unique RGB color.
Apply colormap to the mask: Uses the class index values in the mask to map each pixel to its corresponding color from the colormap.
Blend original image and mask: Combines the original image and the colored mask using cv2.addWeighted() with a 60-40 blending ratio.
Return the blended image: The function returns the blended image, showing the original image with the segmentation mask overlaid.
Image Processing Pipeline:
Define the image path: Specifies the path to the input image.
Preprocess the image: Calls the preprocess() function to load the image, resize it, and convert it to a tensor for the model.
Perform segmentation: Calls segment_image() to get the class-wise segmentation mask.
Overlay the segmentation mask: Calls overlay_segmentation_mask() to blend the mask with the original image.
Display Results:
Set up plot for display: Initializes a figure with two subplots using Matplotlib.
Plot original image: Displays the original image in the first subplot.
Plot segmented image: Displays the image with the segmentation mask in the second subplot.
Show the plots: Calls plt.show() to display both the original and segmented images side by side.
