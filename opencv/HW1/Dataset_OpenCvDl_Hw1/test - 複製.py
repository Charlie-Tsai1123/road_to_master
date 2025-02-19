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

class ImageChannelSeparator(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("Image Channel Separator")
        self.setGeometry(100, 100, 1000, 800)

        # Button to load image
        self.load_button = QPushButton("Load Image", self)
        self.load_button.setGeometry(50, 50, 150, 40)
        self.load_button.clicked.connect(self.load_image)

        # Button for 1-1: Show Color Separation
        self.show_channels_button = QPushButton("1-1: Show Color Separation", self)
        self.show_channels_button.setGeometry(220, 50, 200, 40)
        self.show_channels_button.clicked.connect(self.show_color_channels)
        self.show_channels_button.setEnabled(False)

        # Button for 1-2: Show Grayscale
        self.show_grayscale_button = QPushButton("1-2: Show Grayscale", self)
        self.show_grayscale_button.setGeometry(440, 50, 200, 40)
        self.show_grayscale_button.clicked.connect(self.show_grayscale_images)
        self.show_grayscale_button.setEnabled(False)

        # Button for 1-3: Remove Yellow-Green Colors
        self.remove_yellow_green_button = QPushButton("1-3: Remove Yellow-Green Colors", self)
        self.remove_yellow_green_button.setGeometry(660, 50, 250, 40)
        self.remove_yellow_green_button.clicked.connect(self.remove_yellow_green)
        self.remove_yellow_green_button.setEnabled(False)

        # Button for 2-1: Gaussian Blur
        self.gaussian_blur_button = QPushButton("2.1 Gaussian Blur", self)
        self.gaussian_blur_button.setGeometry(220, 100, 200, 40)
        self.gaussian_blur_button.clicked.connect(self.gaussian_blur_popup)
        self.gaussian_blur_button.setEnabled(False)

        # Button for 2-2: Bilateral Filter
        self.bilateral_filter_button = QPushButton("2.2 Bilateral Filter", self)
        self.bilateral_filter_button.setGeometry(440, 100, 200, 40)
        self.bilateral_filter_button.clicked.connect(self.bilateral_filter_popup)
        self.bilateral_filter_button.setEnabled(False)

        # Button to load image 2
        self.load_button2 = QPushButton("Load Image 2", self)
        self.load_button2.setGeometry(50, 100, 150, 40)
        self.load_button2.clicked.connect(self.load_image2)

        # Button for 2-3: Median Filter
        self.median_filter_button = QPushButton("2.3 Median Filter", self)
        self.median_filter_button.setGeometry(660, 100, 200, 40)
        self.median_filter_button.clicked.connect(self.median_filter_popup)
        self.median_filter_button.setEnabled(False)

        # Button for 3-1: Sobel X
        self.sobel_x_button = QPushButton("3.1: Sobel X", self)
        self.sobel_x_button.setGeometry(50, 150, 150, 40)
        self.sobel_x_button.clicked.connect(self.sobel_x_popup)
        self.sobel_x_button.setEnabled(False)  # Initially disable this button

        # Button for 3.2: Sobel Y
        self.sobel_y_button = QPushButton("3.2: Sobel Y", self)
        self.sobel_y_button.setGeometry(220, 150, 150, 40)
        self.sobel_y_button.clicked.connect(self.sobel_y_popup)
        self.sobel_y_button.setEnabled(False)  # Initially disable this button

        # Button for 3.3: Combination and Threshold
        self.combination_button = QPushButton("3.3: Combination and Threshold", self)
        self.combination_button.setGeometry(400, 150, 250, 40)
        self.combination_button.clicked.connect(self.combination_and_threshold)
        self.combination_button.setEnabled(False)  # Initially disable this button

        # Button for 3.4: Gradient Angle
        self.gradient_angle_button = QPushButton("3.4: Gradient Angle", self)
        self.gradient_angle_button.setGeometry(680, 150, 200, 40)
        self.gradient_angle_button.clicked.connect(self.gradient_angle)
        self.gradient_angle_button.setEnabled(False)  # Initially disable this button

        # Button for 4. Transforms
        self.transform_button = QPushButton("4. Transforms", self)
        self.transform_button.setGeometry(50, 200, 150, 40)
        self.transform_button.clicked.connect(self.show_transform_dialog)
        self.transform_button.setEnabled(False)

        # Placeholder for the loaded image
        self.image = None
        self.channel_windows = []

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

        # 取得輸入框中的數值
        angle = float(self.angle_input.text())  # 旋轉角度（例如 30 度）
        scale = float(self.scale_input.text())  # 縮放比例（例如 0.9）
        trans_x = float(self.trans_x_input.text())  # X 軸平移（例如 775 - 240 = 535）
        trans_y = float(self.trans_y_input.text())  # Y 軸平移（例如 535 - 200 = 335）

        # 設定旋轉和縮放的中心點為 (240, 200)
        (h, w) = self.image.shape[:2]
        center = (240, 200)

        # 計算旋轉矩陣（包含旋轉與縮放），基於指定中心點 (240, 200)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        # 將平移加入到旋轉矩陣中，使圖像中心點從 (240, 200) 移動到新位置 (775, 535)
        rotation_matrix[0, 2] += trans_x
        rotation_matrix[1, 2] += trans_y

        # 套用旋轉、縮放和平移變換
        transformed_image = cv2.warpAffine(self.image, rotation_matrix, (w, h))

        # 顯示結果
        cv2.imshow("Transformed Image", transformed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



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
