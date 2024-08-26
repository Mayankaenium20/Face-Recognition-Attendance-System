# Face Recognition Attendance System

A Python-based attendance system using real-time face recognition with OpenCV and `face_recognition`. This project captures video through a webcam, detects faces, matches them with a pre-trained dataset, and logs attendance with timestamps in a CSV file.

## Features

- **Real-Time Face Detection**: Captures video from a webcam and detects faces using HOG (Histogram of Oriented Gradients).
- **Face Recognition**: Recognizes faces using FaceNet-based embeddings.
- **Attendance Logging**: Automatically marks attendance and logs it with the current date and time.
- **CSV Export**: Saves attendance data in a CSV file for easy record-keeping.

##  Models Used
- **HOG (Histogram of Oriented Gradients)**:
  - **Function**: `face_recognition.face_locations(imgS)`
  - **Usage**: Detects faces in the video frames using the HOG model, which is efficient for face detection.

- **FaceNet**:
  - **Function**: `face_recognition.face_encodings(img)`
  - **Usage**: Generates facial embeddings or encodings that uniquely represent each face, allowing for accurate face recognition and matching.
 
Both HOG and FaceNet are lightweight models, making them ideal for handling light to medium-sized datasets. HOG efficiently detects faces with minimal computational resources, while FaceNet provides accurate face embeddings that are compact and fast to compute. Together, they offer a balanced solution for real-time face recognition tasks, suitable for projects that do not require large-scale data processing or deep learning complexity.

## Installation

### Prerequisites

- Python 3.x
- `pip` package manager

### Setting Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies and avoid conflicts with other projects.

1. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment**:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

3. **Install Dependencies**:
   With the virtual environment activated, install the required Python packages using the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

4. **Deactivating the Virtual Environment**:
   Once you are done working on the project, deactivate the virtual environment:
   ```bash
   deactivate
   ```

### Hardware Requirements

- A webcam or external camera for capturing video.

## Usage

1. **Prepare Training Data**:
   - Place images of the individuals to be recognized in the `TrainingImages` folder.
   - The filenames should reflect the names of the individuals (e.g., `JohnDoe.jpg`).

2. **Run the System**:
   - Start the attendance system by running the main script:
     ```bash
     python main.py
     ```
   - The system will open the webcam feed, detect faces, recognize them, and log the attendance in `Attendance.csv`.

3. **Check Attendance Records**:
   - The attendance records will be saved in the `Attendance.csv` file with columns for name, date, and time.

## Project Structure

```
.
├── TrainingImages/        # Directory containing training images
├── main.py                # Main script to run the attendance system
├── Attendance.csv         # CSV file where attendance is logged
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Future Enhancements

- **Improve Accuracy**: Implement more advanced models like EfficientNet or ResNeXt for better face recognition accuracy.
- **Add GUI**: Develop a graphical user interface (GUI) for easier interaction.
- **Cloud Integration**: Integrate with cloud services for remote attendance tracking and storage.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any feature requests, bugs, or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
