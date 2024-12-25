import openai
import cloudinary
import cloudinary.api
import requests
import os
import gradio as gr
import pytesseract
from PIL import Image
import re
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from transformers import AutoTokenizer, AutoModelForCausalLM

# Cloudinary Configuration
cloudinary.config(
    cloud_name='dopwnz1ze',
    api_key='172931444248964',
    api_secret='x6UFGqc1cSfBWcrTAGy7odv8duA'
)

# Predefined field names for document types
FIELD_NAMES = {
    "Balance Sheet": ["Assets", "Liabilities", "Equity", "Current Assets"],
    "Profit & Loss": ["Revenue", "Expenses", "Net Profit", "Operating Income"],
    "Payslip": ["Employee Name", "Gross Salary", "Deductions", "Net Pay", "Bonus"]
}

# Load GPT-2 model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Function to fetch available folders in Cloudinary
def fetch_cloudinary_folders():
    try:
        response = cloudinary.api.root_folders()
        folders = [folder['name'] for folder in response.get('folders', [])]
        return folders
    except Exception as e:
        print(f"Error fetching folders: {e}")
        return []

def fetch_images_from_cloudinary(folder_name, num_images, save_local=True, local_dir="retrieved_images"):
    image_paths = []
    next_cursor = None
    retrieved_count = 0

    # Ensure the local directory exists
    if save_local and not os.path.exists(local_dir):
        os.makedirs(local_dir)

    while retrieved_count < num_images:
        try:
            response = cloudinary.api.resources(type="upload", prefix=folder_name, max_results=min(100, num_images - retrieved_count), next_cursor=next_cursor)
            images = response.get('resources', [])
            next_cursor = response.get('next_cursor')

            for image in images:
                if retrieved_count >= num_images:
                    break
                image_url = image['secure_url']
                img_data = requests.get(image_url).content

                # Save the image to a temporary file and also save locally
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
                    tmpfile.write(img_data)
                    image_paths.append(tmpfile.name)
                    
                    if save_local:
                        # Save the image to the local directory
                        local_image_path = os.path.join(local_dir, os.path.basename(tmpfile.name))
                        with open(local_image_path, 'wb') as f:
                            f.write(img_data)

                retrieved_count += 1

            if not next_cursor:
                break
        except Exception as e:
            return f"Error retrieving images: {str(e)}", []

    return "Images retrieved successfully!", image_paths


# Function to extract text from an image using OCR
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image, config='--psm 6')
    return text

# Function to extract data based on predefined terms using regex
def extract_data_using_regex(text, document_type):
    extracted_data = {}
    for term in FIELD_NAMES.get(document_type, []):
        pattern = r"\b" + re.escape(term) + r"\b.*?(\d[\d,\.]*)"
        matches = re.search(pattern, text, flags=re.IGNORECASE)
        extracted_data[term] = matches.group(1) if matches else "null"
    return extracted_data

# Function to extract data using GPT-2 (openai-community/gpt2)
def extract_data_using_llm(text, document_type):
    prompt = f"""
    I have extracted the following text from a {document_type}. Please extract the following fields: {', '.join(FIELD_NAMES.get(document_type, []))}.
    Text: {text}
    """
    
    # Add padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as padding token

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    
    # Generate the response from the model using max_new_tokens
    output = model.generate(inputs["input_ids"], max_new_tokens=200)
    
    # Decode the response
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the fields from the decoded output
    extracted_data = {}
    lines = decoded_output.split("\n")
    for line in lines:
        for term in FIELD_NAMES.get(document_type, []):
            if term in line:
                field_value = line.split(":")[-1].strip()
                extracted_data[term] = field_value
    return extracted_data

# Function to clean values (remove commas, convert to float)
def clean_values(values):
    cleaned_values = []
    for value in values:
        value = value.replace(",", "")  # Remove commas
        try:
            cleaned_values.append(float(value))  # Try to convert to float
        except ValueError:
            cleaned_values.append("null")  # If not a valid number, append 'null'
    return cleaned_values

# Function to save images as a ZIP file
def save_images_as_zip(image_paths, zip_name="retrieved_images.zip"):
    zip_path = os.path.join(tempfile.gettempdir(), zip_name)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for img_path in image_paths:
            arcname = os.path.basename(img_path)
            zipf.write(img_path, arcname)
    return zip_path

# Function to extract, store in table format, and visualize
def extract_and_visualize_from_cloudinary(folder_name, num_images, chart_type, document_type):
    # Fetch images from the specified Cloudinary folder
    status, image_paths = fetch_images_from_cloudinary(folder_name, int(num_images))
    if "Error" in status:
        return status, None, None, None, None, None

    extracted_data = {field: [] for field in FIELD_NAMES.get(document_type, [])}
    for img_path in image_paths:
        try:
            image = Image.open(img_path)
            text = extract_text_from_image(image)
            extracted_data_for_image = extract_data_using_regex(text, document_type)  # or extract_data_using_llm
            for field, value in extracted_data_for_image.items():
                extracted_data[field].append(value)
        except Exception as e:
            return f"Error processing image: {str(e)}", None, None, None, None, None

    # Clean data for plotting and table representation
    cleaned_data = {}
    summed_data = {}
    for field, values in extracted_data.items():
        cleaned_values = clean_values(values)  # Now using clean_values
        cleaned_data[field] = cleaned_values
        summed_data[field] = sum([v for v in cleaned_values if isinstance(v, float)])

    # Create a DataFrame for the main data table
    max_length = max(len(v) for v in cleaned_data.values())
    padded_data = {field: values + [""] * (max_length - len(values)) for field, values in cleaned_data.items()}
    df = pd.DataFrame(padded_data)

    # Create a DataFrame for min and max values
    min_max_data = {
        "Field": [],
        "Highest Value": [],
        "Lowest Value": []
    }
    for field, values in cleaned_data.items():
        numeric_values = [v for v in values if isinstance(v, float)]
        min_max_data["Field"].append(field)
        min_max_data["Highest Value"].append(max(numeric_values, default="null"))
        min_max_data["Lowest Value"].append(min(numeric_values, default="null"))

    min_max_df = pd.DataFrame(min_max_data)

    # Save DataFrames as temporary CSV files
    main_csv_path = f"{document_type.replace(' ', '_').lower()}_main.csv"
    min_max_csv_path = f"{document_type.replace(' ', '_').lower()}_min_max.csv"

    df.to_csv(main_csv_path, index=False)
    min_max_df.to_csv(min_max_csv_path, index=False)

    # Save images as ZIP
    zip_file_path = save_images_as_zip(image_paths)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab20.colors

    if chart_type == "Bar Chart":
        ax.bar(summed_data.keys(), summed_data.values(), color=colors[:len(summed_data)])
        ax.set_title(f"{document_type} Data Visualization (Bar Chart)")
        ax.set_ylabel('Total Value')
        ax.set_xlabel('Fields')
        plt.xticks(rotation=45, ha="right")
    elif chart_type == "Pie Chart":
        ax.pie(
            summed_data.values(),
            labels=summed_data.keys(),
            autopct='%1.1f%%',
            startangle=140,
            colors=colors[:len(summed_data)]
        )
        ax.set_title(f"{document_type} Data Visualization (Pie Chart)")

    plt.tight_layout()

    # Save the chart as a temporary image file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        chart_path = tmpfile.name
        plt.savefig(chart_path)

    plt.close()

    return df.to_html(index=False), min_max_df.to_html(index=False), chart_path, main_csv_path, min_max_csv_path, zip_file_path

# Gradio interface
def create_gradio_interface():
    cloudinary_folders = fetch_cloudinary_folders()

    with gr.Blocks() as interface:
        gr.Markdown(""" 
            <div style="text-align: center; font-size: 24px; font-weight: bold;">
                🌥️ Cloudinary Image OCR and Data Visualization 🌥️
            </div>
            <div style="text-align: center; font-size: 16px; color: gray;">
                Select your options below, and visualize extracted data efficiently.
            </div>
        """)

        # Input Section
        gr.Markdown("### Input Options")
        with gr.Row():
            folder_name = gr.Dropdown(cloudinary_folders, label="Select Cloudinary Folder", value=cloudinary_folders[0] if cloudinary_folders else None)
            num_images = gr.Number(value=5, label="Number of Images to Fetch", precision=0)
            chart_type = gr.Dropdown(choices=["Bar Chart", "Pie Chart"], value="Bar Chart", label="Chart Type")
            document_type = gr.Dropdown(choices=["Balance Sheet", "Profit & Loss", "Payslip"], value="Balance Sheet", label="Document Type")

        # Output Section
        gr.Markdown("### Output")
        with gr.Row():
            data_table = gr.HTML(label="Extracted Data Table")
            min_max_table = gr.HTML(label="Min/Max Value Table")
            chart = gr.Image(label="Visualization Chart", type="filepath")

        # Submit button functionality
        submit_btn = gr.Button("Extract & Visualize")
        submit_btn.click(extract_and_visualize_from_cloudinary, inputs=[folder_name, num_images, chart_type, document_type], outputs=[data_table, min_max_table, chart])

    return interface

# Launch the Gradio interface
if __name__ == "__main__":
    create_gradio_interface().launch(debug=True)
