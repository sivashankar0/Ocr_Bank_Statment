# ocr_bank_statment

Cloudinary OCR and Data Visualization
This project allows you to extract text from images using OCR (Optical Character Recognition) and visualize the extracted data from financial documents such as Balance Sheets, Profit & Loss Statements, and Payslips. The data is retrieved from Cloudinary, processed through OCR, and displayed in interactive tables with various visualizations like bar charts and pie charts.

**Features**
Cloudinary Integration: Fetch images stored in Cloudinary folders.
OCR Text Extraction: Use Tesseract OCR to extract text from images.
Data Extraction: Extract relevant data fields from the extracted text based on predefined terms (e.g., Revenue, Expenses, Net Profit).
Data Visualization: Visualize the extracted data in Bar Charts or Pie Charts.
Data Analysis: Generate tables with the highest and lowest values for the extracted data.
Export Options: Save the processed data as CSV files and a ZIP file containing the images.
Prerequisites
Before running the project, make sure you have the following installed:

Python 3.x
Required Python libraries (listed below)
Install Dependencies
bash
Copy code
pip install openai cloudinary requests pytesseract pillow gradio transformers pandas matplotlib
Tesseract Setup
Ensure that Tesseract OCR is installed on your system:

Windows: Download the installer from here and follow the instructions.
Linux (Ubuntu): Run sudo apt-get install tesseract-ocr.
MacOS: Run brew install tesseract.
Once installed, make sure to configure the path to the Tesseract executable (if necessary) in the script.

Project Overview
Folder Structure
main.py: The main script that contains all the functionality for OCR, data extraction, and visualization.
requirements.txt: A file listing all the dependencies.
README.md: The documentation file you are reading.
Functionality
Cloudinary Configuration:

Set up your Cloudinary credentials to interact with the API and fetch images from your Cloudinary account.
Text Extraction from Image:

Images fetched from Cloudinary are processed using Tesseract OCR to extract textual data.
Data Extraction:

Based on the predefined field names for each document type (e.g., "Revenue" for Profit & Loss), the text is parsed and relevant data is extracted using regular expressions or GPT-2 model.
Data Visualization:

The extracted data is visualized in a Bar Chart or Pie Chart, and displayed within a Gradio interface.
Export Data:

The data can be downloaded as CSV files, and a ZIP file of the images can also be generated.
Gradio Interface
This project uses Gradio for a web-based user interface:

Cloudinary Folder Selection: Select the Cloudinary folder where the images are stored.
Image Count: Choose the number of images to retrieve.
Chart Type: Choose between a Bar Chart or Pie Chart to visualize the data.
Document Type: Choose between Balance Sheet, Profit & Loss, or Payslip for data extraction.
Output: The extracted data is displayed in an interactive table format, and the chart visualization is generated.
Example of the Interface
The interface allows you to interact with the application, submit parameters, and get the results instantly:

A data table showing the extracted data.
A table with the highest and lowest values for the fields.
A chart to visualize the data.
How to Use
Clone this repository:

bash
Copy code
git clone https://github.com/your-username/cloudinary-ocr-data-visualization.git
cd cloudinary-ocr-data-visualization
Install the dependencies as mentioned above.

Set up your Cloudinary credentials in the main.py script.

Run the script:

bash
Copy code
python main.py
Open the Gradio interface in your browser, select the Cloudinary folder, specify the parameters (number of images, chart type, document type), and click Extract & Visualize.

The output will be displayed directly on the web interface.

License
This project is licensed under the MIT License - see the LICENSE file for details.
