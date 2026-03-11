# Guided Decision-Making in Machine Learning App

This application is an interactive web dashboard built with **Streamlit** and powered by advanced machine learning libraries including **AutoGluon**, **XGBoost**, **LightGBM**, and **CatBoost**.

To ensure a completely seamless installation experience—without the headache of managing local Python environments or complex package dependencies—this project is fully containerized using **Docker** and **Conda**. 

## 🚀 Prerequisites

Before you begin, ensure you have the following tools installed and running on your machine:
* [Git](https://git-scm.com/downloads) (to download the project)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (to build and run the environment)

---

## 🛠️ Installation and Setup

Follow these simple steps to get the app running on your local machine.

### 1. Clone the repository

First, open your terminal or command prompt and download the project files:
```bash
git clone https://github.com/Amruth-AK/European-Masters-Team-Project.git
```
```
cd European-Masters-Team-Project
```

### 2. Build the Docker Image

Next, build the Docker container. This step reads the Dockerfile and environment.yml to automatically install Python, Miniconda, and all the required machine learning packages.

Run this command (don't forget the period at the end!):
```bash
docker build -t emtp-app .
```
⏳ Note: Because this project installs heavy data science libraries like AutoGluon and CatBoost, the initial build process will take a few minutes. Let it run until it finishes completely.


### 3. Run the Container

Once the build is finished, spin up the app and connect it to your computer's local port:

```bash
docker run -p 8501:8501 emtp-app
```

### 4. View the App

Open your web browser and navigate to:
```
👉 http://localhost:8501
```
Guided Decision-Making in Machine Learning App is running!


---

