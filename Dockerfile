# 1. Use an official Miniconda base image to handle the .yml file
FROM continuumio/miniconda3

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy ONLY the environment.yml first
# (Doing this first saves time on future builds if you only change your code, not your packages)
COPY environment.yml .

# 4. Create the Conda environment named "EMTP" inside the container
RUN conda env create -f environment.yml

# 5. Copy the rest of your project files from your computer into the container
COPY . .

# 6. Expose port 8501 (the default port Streamlit uses to run)
EXPOSE 8501

# 7. Tell Docker how to start your app using the "EMTP" environment
# NOTE: Change "main.py" to whatever your actual Python script is named!
CMD ["conda", "run", "-n", "EMTP", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]