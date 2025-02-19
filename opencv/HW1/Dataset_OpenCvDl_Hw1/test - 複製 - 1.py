import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget,
    QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLineEdit, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator  # Added QDoubleValidator here
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets

class ImageChannelSeparator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hw1 - untitled")
        self.setGeometry(100, 100, 600, 600)
        self.initUI()
    
    def initUI(self):
        # Main Layout
        main_layout = QtWidgets.QGridLayout()
        
        # Section 1: Image Processing
        group_box1 = QtWidgets.QGroupBox("1. Image Processing")
        layout1 = QtWidgets.QVBoxLayout()
        self.button_color_separation = QtWidgets.QPushButton("1.1 Color Separation")
        self.button_color_transformation = QtWidgets.QPushButton("1.2 Color Transformation")
        self.button_color_extraction = QtWidgets.QPushButton("1.3 Color Extraction")
        
        layout1.addWidget(self.button_color_separation)
        layout1.addWidget(self.button_color_transformation)
        layout1.addWidget(self.button_color_extraction)
        group_box1.setLayout(layout1)
        
        # Section 2: Image Smoothing
        group_box2 = QtWidgets.QGroupBox("2. Image Smoothing")
        layout2 = QtWidgets.QVBoxLayout()
        self.button_gaussian_blur = QtWidgets.QPushButton("2.1 Gaussian blur")
        self.button_bilateral_filter = QtWidgets.QPushButton("2.2 Bilateral filter")
        self.button_median_filter = QtWidgets.QPushButton("2.3 Median filter")
        
        layout2.addWidget(self.button_gaussian_blur)
        layout2.addWidget(self.button_bilateral_filter)
        layout2.addWidget(self.button_median_filter)
        group_box2.setLayout(layout2)
        
        # Section 3: Edge Detection
        group_box3 = QtWidgets.QGroupBox("3. Edge Detection")
        layout3 = QtWidgets.QVBoxLayout()
        self.button_sobel_x = QtWidgets.QPushButton("3.1 Sobel X")
        self.button_sobel_y = QtWidgets.QPushButton("3.2 Sobel Y")
        self.button_combination_threshold = QtWidgets.QPushButton("3.3 Combination and Threshold")
        self.button_gradient_angle = QtWidgets.QPushButton("3.4 Gradient Angle")
        
        layout3.addWidget(self.button_sobel_x)
        layout3.addWidget(self.button_sobel_y)
        layout3.addWidget(self.button_combination_threshold)
        layout3.addWidget(self.button_gradient_angle)
        group_box3.setLayout(layout3)
        
        # Load Image Buttons
        self.load_image1_button = QtWidgets.QPushButton("Load Image 1")
        self.load_image2_button = QtWidgets.QPushButton("Load Image 2")
        
        # Section 4: Transforms with inline inputs
        group_box4 = QtWidgets.QGroupBox("4. Transforms")
        layout4 = QtWidgets.QFormLayout()
        self.rotation_input = QtWidgets.QLineEdit()
        self.scaling_input = QtWidgets.QLineEdit()
        self.tx_input = QtWidgets.QLineEdit()
        self.ty_input = QtWidgets.QLineEdit()
        self.button_transform = QtWidgets.QPushButton("4. Transforms")
        
        layout4.addRow("Rotation:", self.rotation_input)
        layout4.addRow("Scaling:", self.scaling_input)
        layout4.addRow("Tx:", self.tx_input)
        layout4.addRow("Ty:", self.ty_input)
        layout4.addWidget(self.button_transform)
        group_box4.setLayout(layout4)
        
        # Arrange sections in grid
        main_layout.addWidget(self.load_image1_button, 0, 0)
        main_layout.addWidget(self.load_image2_button, 1, 0)
        main_layout.addWidget(group_box1, 0, 1)
        main_layout.addWidget(group_box2, 1, 1)
        main_layout.addWidget(group_box3, 2, 1)
        main_layout.addWidget(group_box4, 0, 2, 2, 1)
        
        # Set layout
        self.setLayout(main_layout)
        
        # Connect signals to functionality
        self.load_image1_button.clicked.connect(self.load_image1)
        self.load_image2_button.clicked.connect(self.load_image2)
        self.button_color_separation.clicked.connect(self.color_separation)
        self.button_color_transformation.clicked.connect(self.color_transformation)
        self.button_color_extraction.clicked.connect(self.color_extraction)
        self.button_gaussian_blur.clicked.connect(self.gaussian_blur)
        self.button_bilateral_filter.clicked.connect(self.bilateral_filter)
        self.button_median_filter.clicked.connect(self.median_filter)
        self.button_sobel_x.clicked.connect(self.sobel_x)
        self.button_sobel_y.clicked.connect(self.sobel_y)
        self.button_combination_threshold.clicked.connect(self.combination_threshold)
        self.button_gradient_angle.clicked.connect(self.gradient_angle)
        self.button_transform.clicked.connect(self.apply_transform)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an image file", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is not None:
                print("Image loaded successfully.")
                self.show_channels_button.setEnabled(True)
                self.show_grayscale_button.setEnabled(True)
                self.remove_yellow_green_button.setEnabled(True)
                self.gaussian_blur_button.setEnabled(True)
                self.bilateral_filter_button.setEnabled(True)
                self.sobel_x_button.setEnabled(True)  # Enable Sobel X button
                self.sobel_y_button.setEnabled(True)  # Enable Sobel Y button
                self.combination_button.setEnabled(True)  # Enable Combination and Threshold button
                self.gradient_angle_button.setEnabled(True)  # Enable Gradient Angle button
                self.transform_button.setEnabled(True)

            else:
                print("Failed to load image.")

    def load_image2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an image file", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.image2 = cv2.imread(file_path)
            if self.image2 is not None:
                print("Image 2 loaded successfully.")
                self.median_filter_button.setEnabled(True)

            else:
                print("Failed to load image 2.")


    def show_color_channels(self):
        if self.image is None:
            print("No image loaded.")
            return

        b, g, r = cv2.split(self.image)
        zeros = np.zeros_like(b)

        b_image = cv2.merge([b, zeros, zeros])
        g_image = cv2.merge([zeros, g, zeros])
        r_image = cv2.merge([zeros, zeros, r])

        self.show_image_in_new_window(b_image, "Blue Channel")
        self.show_image_in_new_window(g_image, "Green Channel")
        self.show_image_in_new_window(r_image, "Red Channel")

    def show_grayscale_images(self):
        if self.image is None:
            print("No image loaded.")
            return
        
        cv_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(self.image)
        avg_gray = (b.astype(np.float32) / 3 + g.astype(np.float32) / 3 + r.astype(np.float32) / 3).astype(np.uint8)

        self.show_image_in_new_window(cv_gray, "Grayscale Image")
        self.show_image_in_new_window(avg_gray, "Average Grayscale Image")

    def remove_yellow_green(self):
        if self.image is None:
            print("No image loaded.")
            return
    
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([15, 20, 20])
        upper_bound = np.array([100, 255, 255])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        mask_inverse = cv2.bitwise_not(mask)
        extracted_image = cv2.bitwise_and(self.image, self.image, mask=mask_inverse)

        self.show_image_in_new_window(mask, "Yellow-Green Mask")
        self.show_image_in_new_window(extracted_image, "Image with Yellow-Green Removed")

    def gaussian_blur_popup(self):
        if self.image is None:
            print("No image loaded.")
            return

        # Create a named window for the trackbar
        cv2.namedWindow("Gaussian Blur")

        # Trackbar callback function to update the Gaussian blur
        def apply_blur(m):
            kernel_size = 2 * m + 1
            if m > 0:
                blurred_image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
                cv2.imshow("Gaussian Blur", blurred_image)
            else:
                # Show original image when m is 0
                cv2.imshow("Gaussian Blur", self.image)

        # Create a trackbar with range from 0 to 5 (0 will show the original image)
        cv2.createTrackbar("Kernel Radius", "Gaussian Blur", 0, 5, apply_blur)

        # Show the original image initially
        apply_blur(0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def bilateral_filter_popup(self):
        if self.image is None:
            print("No image loaded.")
            return

        # Create a named window for the trackbar
        cv2.namedWindow("Bilateral Filter")

        # Set constant sigmaColor and sigmaSpace values
        sigmaColor = 90
        sigmaSpace = 90

        # Trackbar callback function to update the bilateral filter
        def apply_bilateral(m):
            d = 2 * m + 1  # Kernel diameter
            if m > 0:
                bilateral_filtered = cv2.bilateralFilter(self.image, d, sigmaColor, sigmaSpace)
                cv2.imshow("Bilateral Filter", bilateral_filtered)
            else:
                # Show original image when m is 0
                cv2.imshow("Bilateral Filter", self.image)

        # Create a trackbar with range from 0 to 5 (0 will show the original image)
        cv2.createTrackbar("Kernel Radius", "Bilateral Filter", 0, 5, apply_bilateral)

        # Show the original image initially
        apply_bilateral(0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def median_filter_popup(self):
        if self.image2 is None:
            print("No image 2 loaded.")
            return

        # Create a named window for the trackbar
        cv2.namedWindow("Median Filter")

        # Trackbar callback function to update the median filter
        def apply_median(m):
            kernel_size = 2 * m + 1  # Kernel size must be odd
            if m > 0:
                median_filtered = cv2.medianBlur(self.image2, kernel_size)
                cv2.imshow("Median Filter", median_filtered)
            else:
                # Show original image when m is 0
                cv2.imshow("Median Filter", self.image2)

        # Create a trackbar with range from 0 to 5 (0 will show the original image)
        cv2.createTrackbar("Kernel Radius", "Median Filter", 0, 5, apply_median)

        # Show the original image initially
        apply_median(0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sobel_x_popup(self):
        if self.image is None:
            print("No image loaded.")
            return

        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur with small kernel and sigma
        kernel_size = 3
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0.5, 0.5)

        # Define the Sobel X operator
        sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

        # Initialize output image
        height, width = blur.shape
        sobel_x_result = np.zeros_like(blur, dtype=np.float32)

        # Apply Sobel X operator manually
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                region = blur[i-1:i+2, j-1:j+2].astype(np.float32)
                value = np.sum(region * sobel_x)
                sobel_x_result[i, j] = abs(value)

        # Scale the result to preserve more gray levels
        # First normalize to 0-1 range
        sobel_x_result = sobel_x_result / np.max(sobel_x_result)
    
        # Apply non-linear scaling to enhance edges while keeping gradients
        sobel_x_result = np.power(sobel_x_result, 0.7) * 255
    
        # Convert to uint8
        sobel_x_result = sobel_x_result.astype(np.uint8)

        # Skip binary threshold to preserve grayscale levels
        # Instead, use a very low threshold to remove noise
        sobel_x_result[sobel_x_result < 15] = 0

        # Show result
        cv2.imshow("Sobel X Result", sobel_x_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sobel_y_popup(self):
        if self.image is None:
            print("No image loaded.")
            return

        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur with small kernel and sigma
        kernel_size = 3
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0.5, 0.5)

        # Define the Sobel Y operator
        sobel_y = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

        # Initialize output image
        height, width = blur.shape
        sobel_y_result = np.zeros_like(blur, dtype=np.float32)

        # Apply Sobel Y operator manually
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                region = blur[i-1:i+2, j-1:j+2].astype(np.float32)
                value = np.sum(region * sobel_y)
                sobel_y_result[i, j] = abs(value)

        # Scale the result to preserve more gray levels
        # First normalize to 0-1 range
        sobel_y_result = sobel_y_result / np.max(sobel_y_result)
    
        # Apply non-linear scaling to enhance edges while keeping gradients
        sobel_y_result = np.power(sobel_y_result, 0.7) * 255
    
        # Convert to uint8
        sobel_y_result = sobel_y_result.astype(np.uint8)

        # Skip binary threshold to preserve grayscale levels
        # Instead, use a very low threshold to remove noise
        sobel_y_result[sobel_y_result < 15] = 0

        # Show result
        cv2.imshow("Sobel Y Result", sobel_y_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def combination_and_threshold(self):
        if self.image is None:
            print("No image loaded.")
            return

        # Step 1: Convert the image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Step 2: Compute Sobel X and Sobel Y on the grayscale image
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Step 3: Compute the magnitude of the gradient
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Step 4: Normalize the result to 0-255
        normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Step 5: Apply thresholding
        _, result1 = cv2.threshold(normalized, 128, 255, cv2.THRESH_BINARY)
        _, result2 = cv2.threshold(normalized, 28, 255, cv2.THRESH_BINARY)

        # Step 6: Show results
        cv2.imshow("Combined Sobel Result", normalized)
        cv2.imshow("Threshold Result (128)", result1)
        cv2.imshow("Threshold Result (28)", result2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def gradient_angle(self):
        if self.image is None:
            print("No image loaded.")
            return

        # Step 1: Convert the image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Step 2: Compute Sobel X and Sobel Y on the grayscale image
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Step 3: Calculate gradient angle θ
        gradient_angle = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)  # Convert to degrees
        gradient_angle[gradient_angle < 0] += 360  # Ensure all angles are positive

        # Step 4: Generate masks based on the specified angle ranges
        mask1 = np.zeros_like(gradient_angle, dtype=np.uint8)
        mask2 = np.zeros_like(gradient_angle, dtype=np.uint8)

        # Mask for range 170° ~ 190°
        mask1[(gradient_angle >= 170) & (gradient_angle <= 190)] = 255

        # Mask for range 260° ~ 280°
        mask2[(gradient_angle >= 260) & (gradient_angle <= 280)] = 255

        # Step 5: Generate results using cv2.bitwise_and
        combined_result = cv2.normalize(np.sqrt(sobel_x**2 + sobel_y**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        result1 = cv2.bitwise_and(combined_result, mask1)
        result2 = cv2.bitwise_and(combined_result, mask2)

        # Step 6: Show results
        cv2.imshow("Gradient Angle Result (170° ~ 190°)", result1)
        cv2.imshow("Gradient Angle Result (260° ~ 280°)", result2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_transform_dialog(self):
        if self.image is None:
            print("No image loaded.")
            return

        self.transform_dialog = QWidget()
        self.transform_dialog.setWindowTitle("Image Transform")
        self.transform_dialog.setGeometry(200, 200, 400, 300)

        layout = QVBoxLayout()

        # Create input fields
        angle_group = QGroupBox("Rotation")
        angle_layout = QHBoxLayout()
        self.angle_input = QLineEdit()
        self.angle_input.setValidator(QDoubleValidator())
        self.angle_input.setText("30")  # Default value
        angle_layout.addWidget(QLabel("Angle (degrees):"))
        angle_layout.addWidget(self.angle_input)
        angle_group.setLayout(angle_layout)

        scale_group = QGroupBox("Scale")
        scale_layout = QHBoxLayout()
        self.scale_input = QLineEdit()
        self.scale_input.setValidator(QDoubleValidator())
        self.scale_input.setText("0.9")  # Default value
        scale_layout.addWidget(QLabel("Scale factor:"))
        scale_layout.addWidget(self.scale_input)
        scale_group.setLayout(scale_layout)

        trans_group = QGroupBox("Translation")
        trans_layout = QHBoxLayout()
        self.trans_x_input = QLineEdit()
        self.trans_y_input = QLineEdit()
        self.trans_x_input.setText("535")  # Default value
        self.trans_y_input.setText("335")  # Default value
        trans_layout.addWidget(QLabel("X:"))
        trans_layout.addWidget(self.trans_x_input)
        trans_layout.addWidget(QLabel("Y:"))
        trans_layout.addWidget(self.trans_y_input)
        trans_group.setLayout(trans_layout)

        # Add Apply button
        apply_button = QPushButton("Apply Transform")
        apply_button.clicked.connect(self.apply_transform)

        # Add all widgets to main layout
        layout.addWidget(angle_group)
        layout.addWidget(scale_group)
        layout.addWidget(trans_group)
        layout.addWidget(apply_button)

        self.transform_dialog.setLayout(layout)
        self.transform_dialog.show()

    def apply_transform(self):
        if self.image is None:
            print("No image loaded.")
            return
        
        try:
            # Parse the user inputs for transformation parameters
            rotation = float(self.rotation_input.text())
            scaling = float(self.scaling_input.text())
            tx = float(self.tx_input.text())
            ty = float(self.ty_input.text())
            
            # Get the dimensions of the image
            h, w = self.image.shape[:2]
            
            # Calculate the center of the image for rotation
            center = (w // 2, h // 2)
            
            # Apply scaling and rotation transformations
            transform_matrix = cv2.getRotationMatrix2D(center, rotation, scaling)
            
            # Add translation values to the transformation matrix
            transform_matrix[0, 2] += tx
            transform_matrix[1, 2] += ty
            
            # Perform the affine transformation
            transformed_image = cv2.warpAffine(self.image, transform_matrix, (w, h))
            
            # Display the transformed image
            cv2.imshow("Transformed Image", transformed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        except ValueError:
            print("Please enter valid numerical values for all transformation parameters.")




    def show_image_in_new_window(self, img, title):
        window = QWidget()
        window.setWindowTitle(title)

        if len(img.shape) == 2:
            q_img = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_img = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format_RGB888)

        img_label = QLabel()
        img_label.setPixmap(QPixmap.fromImage(q_img).scaled(300, 200, Qt.KeepAspectRatio))
        img_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(img_label)

        window.setLayout(layout)
        window.resize(300, 200)
        window.show()

        self.channel_windows.append(window)

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageChannelSeparator()
    window.show()
    sys.exit(app.exec_())
