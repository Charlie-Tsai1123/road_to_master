import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget,
    QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLineEdit, QGroupBox, QGridLayout, QSpacerItem, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator
from PyQt5.QtCore import Qt

class ImageChannelSeparator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hw1")
        self.setGeometry(100, 100, 600, 700)  # Made window taller and slightly narrower

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # Create left buttons container with center alignment
        buttons_container = QWidget()
        buttons_layout = QVBoxLayout()
        buttons_container.setLayout(buttons_layout)
        buttons_container.setFixedWidth(120)  # Reduced width

        # Add Load Image buttons with center alignment
        self.load_image1_btn = QPushButton("Load Image 1")
        self.load_image2_btn = QPushButton("Load Image 2")
        buttons_layout.addStretch()  # Add stretch before buttons
        buttons_layout.addWidget(self.load_image1_btn, alignment=Qt.AlignCenter)

        # Add spacer between buttons
        spacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        buttons_layout.addItem(spacer)

        buttons_layout.addWidget(self.load_image2_btn, alignment=Qt.AlignCenter)
        buttons_layout.addStretch()  # Add stretch after buttons

        # Create left and right section containers
        left_container = QWidget()
        right_container = QWidget()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        left_container.setLayout(left_layout)
        right_container.setLayout(right_layout)

        # Set fixed widths for containers
        left_container.setFixedWidth(250)  # Reduced width
        right_container.setFixedWidth(200)  # Reduced width

        # Create group boxes with reduced heights
        image_processing_group = QGroupBox("1. Image Processing")
        image_smoothing_group = QGroupBox("2. Image Smoothing")
        edge_detection_group = QGroupBox("3. Edge Detection")

        # Set minimum heights for more compact spacing
        image_processing_group.setMinimumHeight(200)
        image_smoothing_group.setMinimumHeight(200)
        edge_detection_group.setMinimumHeight(260)

        # Create layouts for each group
        image_processing_layout = QVBoxLayout()
        image_smoothing_layout = QVBoxLayout()
        edge_detection_layout = QVBoxLayout()

        # Reduce spacing in layouts
        image_processing_layout.setSpacing(2)
        image_smoothing_layout.setSpacing(2)
        edge_detection_layout.setSpacing(2)

        # Image Processing buttons
        self.color_separation_btn = QPushButton("1.1 Color Separation")
        self.color_transformation_btn = QPushButton("1.2 Color Transformation")
        self.color_extraction_btn = QPushButton("1.3 Color Extraction")
        image_processing_layout.addWidget(self.color_separation_btn)
        image_processing_layout.addWidget(self.color_transformation_btn)
        image_processing_layout.addWidget(self.color_extraction_btn)
        image_processing_group.setLayout(image_processing_layout)

        # Image Smoothing buttons
        self.gaussian_blur_btn = QPushButton("2.1 Gaussian blur")
        self.bilateral_filter_btn = QPushButton("2.2 Bilateral filter")
        self.median_filter_btn = QPushButton("2.3 Median filter")
        image_smoothing_layout.addWidget(self.gaussian_blur_btn)
        image_smoothing_layout.addWidget(self.bilateral_filter_btn)
        image_smoothing_layout.addWidget(self.median_filter_btn)
        image_smoothing_group.setLayout(image_smoothing_layout)

        # Edge Detection buttons
        self.sobel_x_btn = QPushButton("3.1 Sobel X")
        self.sobel_y_btn = QPushButton("3.2 Sobel Y")
        self.combination_btn = QPushButton("3.3 Combination and\nThreshold")
        self.gradient_angle_btn = QPushButton("3.4 Gradient Angle")
        edge_detection_layout.addWidget(self.sobel_x_btn)
        edge_detection_layout.addWidget(self.sobel_y_btn)
        edge_detection_layout.addWidget(self.combination_btn)
        edge_detection_layout.addWidget(self.gradient_angle_btn)
        edge_detection_group.setLayout(edge_detection_layout)

        # Transforms controls
        transforms_group = QGroupBox("4. Transforms")
        transforms_layout = QGridLayout()

        # Set stretches for columns to distribute space evenly
        transforms_layout.setColumnStretch(0, 1)  # Label column
        transforms_layout.setColumnStretch(1, 1)  # Input column
        transforms_layout.setColumnStretch(4, 1)  # Unit label column

        # Make line edits smaller
        line_edit_width = 70
        label_width = 120

        # Add controls with adjusted sizes
        rotation_label = QLabel("Rotation:")
        rotation_label.setFixedWidth(label_width)
        self.rotation_input = QLineEdit()
        self.rotation_input.setFixedWidth(line_edit_width)
        deg_label = QLabel("deg")
        transforms_layout.addWidget(rotation_label, 0, 0)
        transforms_layout.addWidget(self.rotation_input, 0, 1)
        transforms_layout.addWidget(deg_label, 0, 4)

        scaling_label = QLabel("Scaling:")
        scaling_label.setFixedWidth(label_width)
        self.scaling_input = QLineEdit()
        self.scaling_input.setFixedWidth(line_edit_width)
        transforms_layout.addWidget(scaling_label, 1, 0)
        transforms_layout.addWidget(self.scaling_input, 1, 1)

        tx_label = QLabel("Tx:")
        tx_label.setFixedWidth(label_width)
        self.tx_input = QLineEdit()
        self.tx_input.setFixedWidth(line_edit_width)
        pixel_label1 = QLabel("pixel")
        transforms_layout.addWidget(tx_label, 2, 0)
        transforms_layout.addWidget(self.tx_input, 2, 1)
        transforms_layout.addWidget(pixel_label1, 2, 4)

        ty_label = QLabel("Ty:")
        ty_label.setFixedWidth(label_width)
        self.ty_input = QLineEdit()
        self.ty_input.setFixedWidth(line_edit_width)
        pixel_label2 = QLabel("pixel")
        transforms_layout.addWidget(ty_label, 3, 0)
        transforms_layout.addWidget(self.ty_input, 3, 1)
        transforms_layout.addWidget(pixel_label2, 3, 4)

        self.transform_btn = QPushButton("4. Transform")
        transforms_layout.addWidget(self.transform_btn, 4, 0, 1, 3)

        # Set stretch for rows (optional)
        transforms_layout.setRowStretch(0, 1)
        transforms_layout.setRowStretch(1, 1)
        transforms_layout.setRowStretch(2, 1)
        transforms_layout.setRowStretch(3, 1)
        transforms_layout.setRowStretch(4, 1)

        # Add groups to layouts
        transforms_group.setLayout(transforms_layout)

        left_layout.addWidget(image_processing_group)
        left_layout.addWidget(image_smoothing_group)
        left_layout.addWidget(edge_detection_group)
        left_layout.addStretch()

        right_layout.addWidget(transforms_group)
        right_layout.addStretch()

        # Add containers to main layout
        main_layout.addWidget(buttons_container)
        main_layout.addWidget(left_container)
        main_layout.addWidget(right_container)

        # Connect buttons and initialize other components
        self.connect_buttons()
        self.initialize_components()
        
    def connect_buttons(self):
        self.load_image1_btn.clicked.connect(self.load_image)
        self.load_image2_btn.clicked.connect(self.load_image2)
        self.color_separation_btn.clicked.connect(self.show_color_channels)
        self.color_transformation_btn.clicked.connect(self.show_grayscale_images)
        self.color_extraction_btn.clicked.connect(self.remove_yellow_green)
        self.gaussian_blur_btn.clicked.connect(self.gaussian_blur_popup)
        self.bilateral_filter_btn.clicked.connect(self.bilateral_filter_popup)
        self.median_filter_btn.clicked.connect(self.median_filter_popup)
        self.sobel_x_btn.clicked.connect(self.sobel_x_popup)
        self.sobel_y_btn.clicked.connect(self.sobel_y_popup)
        self.combination_btn.clicked.connect(self.combination_and_threshold)
        self.gradient_angle_btn.clicked.connect(self.gradient_angle)
        self.transform_btn.clicked.connect(self.apply_transform)

    def initialize_components(self):
        self.image = None
        self.image2 = None
        self.channel_windows = []
        self.disable_buttons()

    def disable_buttons(self):
        """Disable all processing buttons initially"""
        for btn in [self.color_separation_btn, self.color_transformation_btn,
                   self.color_extraction_btn, self.gaussian_blur_btn,
                   self.bilateral_filter_btn, self.median_filter_btn,
                   self.sobel_x_btn, self.sobel_y_btn, self.combination_btn,
                   self.gradient_angle_btn, self.transform_btn]:
            btn.setEnabled(False)

    # The rest of your methods remain the same, just remove the show_transform_dialog method
    # and modify apply_transform to use the main window inputs

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an image file", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is not None:
                print("Image loaded successfully.")
                for btn in [self.color_separation_btn, self.color_transformation_btn,
                          self.color_extraction_btn, self.gaussian_blur_btn,
                          self.bilateral_filter_btn, self.sobel_x_btn,
                          self.sobel_y_btn, self.combination_btn,
                          self.gradient_angle_btn, self.transform_btn]:
                    btn.setEnabled(True)
            else:
                print("Failed to load image.")

    def load_image2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an image file", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.image2 = cv2.imread(file_path)
            if self.image2 is not None:
                print("Image 2 loaded successfully.")
                # Enable only the median filter button since it's the only one that uses image2
                self.median_filter_btn.setEnabled(True)
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

        # Convert the image to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0.5, 0.5)

        # Compute Sobel X and Y manually
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        height, width = blur.shape
        sobel_x_result = np.zeros_like(blur, dtype=np.float32)
        sobel_y_result = np.zeros_like(blur, dtype=np.float32)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                region = blur[i-1:i+2, j-1:j+2].astype(np.float32)
                sobel_x_result[i, j] = np.sum(region * sobel_x)
                sobel_y_result[i, j] = np.sum(region * sobel_y)

        # Calculate gradient magnitude
        magnitude = np.sqrt(sobel_x_result**2 + sobel_y_result**2)
        normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply thresholds
        _, result1 = cv2.threshold(normalized, 128, 255, cv2.THRESH_BINARY)
        _, result2 = cv2.threshold(normalized, 28, 255, cv2.THRESH_BINARY)

        cv2.imshow("Combined Sobel Result", normalized)
        cv2.imshow("Threshold Result (128)", result1)
        cv2.imshow("Threshold Result (28)", result2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def gradient_angle(self):
        if self.image is None:
            print("No image loaded.")
            return

        # Convert the image to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0.5, 0.5)

        # Compute Sobel X and Y manually
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        height, width = blur.shape
        sobel_x_result = np.zeros_like(blur, dtype=np.float32)
        sobel_y_result = np.zeros_like(blur, dtype=np.float32)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                region = blur[i-1:i+2, j-1:j+2].astype(np.float32)
                sobel_x_result[i, j] = np.sum(region * sobel_x)
                sobel_y_result[i, j] = np.sum(region * sobel_y)

        # Calculate gradient angle
        gradient_angle = np.arctan2(sobel_y_result, sobel_x_result) * (180 / np.pi)
        gradient_angle[gradient_angle < 0] += 360

        # Create masks based on angle ranges
        mask1 = np.zeros_like(gradient_angle, dtype=np.uint8)
        mask2 = np.zeros_like(gradient_angle, dtype=np.uint8)
        mask1[(gradient_angle >= 170) & (gradient_angle <= 190)] = 255
        mask2[(gradient_angle >= 260) & (gradient_angle <= 280)] = 255

        # Gradient magnitude (combined result)
        combined_result = cv2.normalize(np.sqrt(sobel_x_result**2 + sobel_y_result**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply masks to combined result
        result1 = cv2.bitwise_and(combined_result, mask1)
        result2 = cv2.bitwise_and(combined_result, mask2)

        cv2.imshow("Gradient Angle Result (170° ~ 190°)", result1)
        cv2.imshow("Gradient Angle Result (260° ~ 280°)", result2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def apply_transform(self):
        if self.image is None:
            print("No image loaded.")
            return

        try:
            angle = float(self.rotation_input.text() or "0")
            scale = float(self.scaling_input.text() or "1")
            trans_x = float(self.tx_input.text() or "0")
            trans_y = float(self.ty_input.text() or "0")

            (h, w) = self.image.shape[:2]
            center = (240, 200)

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            rotation_matrix[0, 2] += trans_x
            rotation_matrix[1, 2] += trans_y

            transformed_image = cv2.warpAffine(self.image, rotation_matrix, (w, h))

            cv2.imshow("Transformed Image", transformed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except ValueError:
            print("Please enter valid numbers for transformation parameters")



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
