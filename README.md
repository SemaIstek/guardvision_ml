
# GuardVision ML

GuardVision ML is an image processing and machine learning-based analysis system. This project is developed to perform object detection and analysis on images taken from security cameras.

## Features
- YOLO-based object detection
- Model training and test files
- Visualization and analysis with Jupyter Notebook
- Python virtual environment support

## Installation
1. Python 3.10+ must be installed.
2. Create a virtual environment:
	```bash
	python3 -m venv venv
	source venv/bin/activate
	```
3. Install required packages:
	```bash
	pip install -r requirements.txt
	```

## Usage
- `main.py`: Main application file
- `test/test_ml.py`: Test file
- `YOLOv11_union.ipynb`: Analysis and visualization with notebook

## File Structure
- `images/`: Test images
- `guardvision_ml/`: Main project folder
- `best.pt`: Trained model file
- `training_summary.pkl`: Training summary

## Contribution
To contribute, please send a pull request or open an issue.

## License
This project is licensed under the MIT License.