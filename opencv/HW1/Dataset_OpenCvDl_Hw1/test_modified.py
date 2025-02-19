
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget,
    QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QGroupBox, QGridLayout
)
from PyQt5.QtGui import QDoubleValidator  # Validator for floating-point inputs

class ImageChannelSeparator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hw1 - untitled")
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget and layout
        main_widget = QWidget(self)
        main_layout = QGridLayout(main_widget)
        
        # Section 1: Image Processing
        group1 = QGroupBox("1. Image Processing")
        group1_layout = QVBoxLayout()
        group1.setLayout(group1_layout)
        
        self.color_separation_button = QPushButton("1.1 Color Separation")
        self.color_transformation_button = QPushButton("1.2 Color Transformation")
        self.color_extraction_button = QPushButton("1.3 Color Extraction")
        
        group1_layout.addWidget(self.color_separation_button)
        group1_layout.addWidget(self.color_transformation_button)
        group1_layout.addWidget(self.color_extraction_button)

        # Section 2: Image Smoothing
        group2 = QGroupBox("2. Image Smoothing")
        group2_layout = QVBoxLayout()
        group2.setLayout(group2_layout)
        
        self.gaussian_blur_button = QPushButton("2.1 Gaussian blur")
        self.bilateral_filter_button = QPushButton("2.2 Bilateral filter")
        self.median_filter_button = QPushButton("2.3 Median filter")
        
        group2_layout.addWidget(self.gaussian_blur_button)
        group2_layout.addWidget(self.bilateral_filter_button)
        group2_layout.addWidget(self.median_filter_button)
        
        # Section 3: Edge Detection
        group3 = QGroupBox("3. Edge Detection")
        group3_layout = QVBoxLayout()
        group3.setLayout(group3_layout)
        
        self.sobel_x_button = QPushButton("3.1 Sobel X")
        self.sobel_y_button = QPushButton("3.2 Sobel Y")
        self.combination_threshold_button = QPushButton("3.3 Combination and Threshold")
        self.gradient_angle_button = QPushButton("3.4 Gradient Angle")
        
        group3_layout.addWidget(self.sobel_x_button)
        group3_layout.addWidget(self.sobel_y_button)
        group3_layout.addWidget(self.combination_threshold_button)
        group3_layout.addWidget(self.gradient_angle_button)
        
        # Section 4: Transforms (with inline input fields)
        group4 = QGroupBox("4. Transforms")
        group4_layout = QFormLayout()
        group4.setLayout(group4_layout)
        
        self.rotation_input = QLineEdit()
        self.rotation_input.setPlaceholderText("deg")
        
        self.scaling_input = QLineEdit()
        self.scaling_input.setPlaceholderText("")
        
        self.tx_input = QLineEdit()
        self.tx_input.setPlaceholderText("pixel")
        
        self.ty_input = QLineEdit()
        self.ty_input.setPlaceholderText("pixel")
        
        # Adding input fields for transforms
        group4_layout.addRow("Rotation:", self.rotation_input)
        group4_layout.addRow("Scaling:", self.scaling_input)
        group4_layout.addRow("Tx:", self.tx_input)
        group4_layout.addRow("Ty:", self.ty_input)
        
        self.transform_button = QPushButton("4. Transforms")
        group4_layout.addWidget(self.transform_button)

        # Load Image Buttons
        load_button1 = QPushButton("Load Image 1")
        load_button2 = QPushButton("Load Image 2")
        
        # Adding everything to the main layout
        main_layout.addWidget(load_button1, 0, 0)
        main_layout.addWidget(load_button2, 1, 0)
        main_layout.addWidget(group1, 0, 1, 2, 1)
        main_layout.addWidget(group2, 2, 1)
        main_layout.addWidget(group3, 3, 1)
        main_layout.addWidget(group4, 0, 2, 4, 1)

        self.setCentralWidget(main_widget)

app = QApplication(sys.argv)
window = ImageChannelSeparator()
window.show()
sys.exit(app.exec_())
