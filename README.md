
# 🌟 Star Child Wellness Platform

## Description

The **Star** platform is a comprehensive system designed to help mothers monitor their child's emotional and nutritional state. It offers a range of features through different interconnected modules that analyze multiple types of input, including text and images. The platform provides detailed analysis, reports, and personalized advice based on the data processed. By integrating advanced AI models and computer vision techniques, it analyzes drawings, processes food images, and enables mothers to connect with experts for professional guidance and support.


## Features

### 1. Authentication Module

* **Personalized Accounts:** The website enables mothers to create and manage personal accounts, including login and sign-up functionality.
* **Data & History:** After registration, the mother has a profile where data and historical actions are saved, such as her child's personal data, historical child state, and past reports.

### 2. Home Page Module

* **Central Hub:** The home page displays all the key features available on the platform.
* **Easy Access:** Mothers can easily access sections related to mental health status, nutrition tracking, recent reports, and messages with experts.

### 3. Mental Health Analysis Module

* **Drawings Analysis:** Mothers can upload their child's drawings. The system then analyzes colors, shapes, and lines to identify the child's behavior and emotional state. All outputs are integrated into a PDF report.
* **AI Chatbot:** Mothers can interact with a chatbot to ask questions or describe their child's behavior. The chatbot provides personalized answers, advice on how to deal with the child, or an initial mental health assessment.

### 4. Nutrition Analysis Module

* **Nutrition Dashboard:** This feature provides a summary of historical nutritional data, including the amount of calories per day, macronutrient distribution, and nutritional tendencies. Based on this, the platform provides personalized recommendations to ensure balanced eating habits.
* **Food Classifier:** Mothers can upload images of meals. A CNN-based food recognition model identifies the food items present and estimates the calories, carbohydrates, and nutrients.
* **AI Chatbot:** A dedicated chatbot answers nutritional questions, provides guidance, and suggests balanced, healthy recipes based on available ingredients.

### 5. Scoring Module

* **Overall Health Score:** Each analysis (mental and nutritional) contributes to an overall health score that summarizes the general state of the child.
* **Component Scores:** This includes a **mental health score** derived from emotional tracking over time and a **nutrition score** calculated based on historical calorie balance and meal diversity.

### 6. Expert Contact and Messaging Module

* **Professional Access:** Mothers can directly contact health or mental health experts through the platform.
* **Informed Consultation:** This allows them to discuss the AI-generated analysis and reports to receive professional advice about their child's situation.

---

## How It Works: Data Flow

The platform's data flow proceeds as follows:

1.  **User Authentication:** The mother logs in to access her profile. Authentication is verified via the database.
2.  **Input Submission:** The user submits various inputs, such as uploading a child's photo or drawing, uploading a food photo, or entering a text description.
3.  **Backend Routing:** The input is sent to a Flask API, which determines the appropriate destination, sending texts to chatbot models and images to AI models (like the drawing analyzer or food classifier).
4.  **Data Preprocessing:** Input data passes through a preprocessing phase. Images are resized, normalized, and cleaned using PIL and NumPy. Textual data is tokenized and embedded for language models.
5.  **AI Processing and Analysis:**
    * **Mental Health:** Text is processed by an RAG chatbot (using a LLaMA-based model) which retrieves information from a medical book. The Drawing Analysis System combines models like ViT (Emotion Classification), DETR (Object Detection), and BLIP-2 (Image captioning).
    * **Nutrition:** Food images are analyzed in two phases. First, YOLO detects food items. Second, a TensorFlow CNN classifies the food type and links it to nutritional data. The nutrition chatbot uses a fine-tuned LLaMa 3B model.
6.  **Data Storage:** All outputs, such as emotion classification, nutritional values, and conversation logs, are transmitted through the Flask API and stored in the backend database.
7.  **Visualization and Reporting:** The stored results are dynamically retrieved to be displayed on dashboards showing historical analysis or in automatically generated PDF reports using pyPDF2.

---

## 🛠️ Technology Stack

The platform is built using a robust architecture and modern AI models:

* **Backend:** Flask API, Fast API
* **AI / LLM:** LLaMA (via Groq API), Fine-tuned LLaMa 3B, RAG (Retrieval-Augmented Generation)
* **Drawing Analysis (CV):** ViT (Vision Transformer), DETR (Object Detection), BLIP-2 (Image captioning)
* **Food Analysis (CV):** YOLO (Object Detection), TensorFlow (CNN)
* **Data Handling/Preprocessing:** Pandas, NumPy, PIL, OpenCV, TRL (Transformer Reinforcement Learning)
* **Report Generation:** pyPDF2
