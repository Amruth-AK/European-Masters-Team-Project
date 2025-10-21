# Project Setup Guide

## Installation

### 1. Install Python Dependencies

```bash
pip install pandas numpy
```

Or create a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install pandas numpy
```

---

## File Structure

```
European-Masters-Team-Project/
├── main.py                          # Main integration script
├── analyze.py                       # Analysis module (Adrian & Alisa)
├── preprocessing_suggestions.py     # Suggestion engine (Yang, Amruth, Tecla)
├── mock_analysis_output.py          # Mock data for testing
├── DATA_CONTRACT.md                 # Interface documentation
├── SETUP_GUIDE.md                   # This file
│
└── Preprocessing suggestion/
    ├── preprocessing_function.py    # Preprocessing functions library
    ├── Preprocessing_suggestion Outline.md
    └── ... (other related files)
```

---

## Usage

### Demo Mode (No Dataset Required)

```bash
python3 main.py
```

This will:
- Use mock analysis data
- Generate preprocessing suggestions
- Display them in the terminal

### With Actual Dataset

```bash
python3 main.py path/to/dataset.csv TargetColumn
```

Example with Titanic dataset:
```bash
python3 main.py data/titanic.csv Survived
```

### With Output Path

```bash
python3 main.py input.csv Target output_preprocessed.csv
```

---

## Testing Individual Modules

### Test Mock Data

```bash
cd European-Masters-Team-Project
python3 mock_analysis_output.py
```

### Test Preprocessing Suggestions

```bash
python3 preprocessing_suggestions.py
```

### Test Preprocessing Functions

```bash
cd "Preprocessing suggestion"
python3 -c "from preprocessing_function import *; print('Functions loaded successfully')"
```

---

## Development Workflow

### For Team Members

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd European-Masters-Team-Project
   ```

2. **Create your branch** (if not already done)
   ```bash
   git checkout -b your-name
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy
   ```

4. **Implement your functions**
   - Find your section in the relevant file
   - Look for `TODO: [Your Name]` comments
   - Follow the function signature pattern

5. **Test your code**
   ```bash
   python3 main.py  # Test integration
   ```

6. **Commit and push**
   ```bash
   git add .
   git commit -m "Implement [feature name]"
   git push origin your-name
   ```

---

## Current Implementation Status

### ✅ Complete

- Project structure
- Main integration pipeline
- Mock data for testing
- Data contract documentation
- Numerical scaling functions (Yang)
- Outlier handling functions (Yang)

### ⏳ In Progress

- Analysis module (Adrian & Alisa)
- Categorical encoding functions (Amruth)
- Missing value handling (Tecla)
- Duplicate handling (Tecla)

---

## Troubleshooting

### ModuleNotFoundError: No module named 'pandas'

**Solution:**
```bash
pip install pandas numpy
```

### ImportError: cannot import preprocessing functions

**Solution:** Make sure you're in the correct directory:
```bash
cd European-Masters-Team-Project
python3 main.py
```

### Git push rejected

**Solution:** Pull latest changes first:
```bash
git pull origin your-branch
git push origin your-branch
```

---

## Dependencies

- Python 3.7+
- pandas
- numpy

---

## Contact

For questions about:
- **Analysis module**: Contact Adrian or Alisa
- **Numerical features & outliers**: Contact Yang/Zhiqi
- **Categorical encoding**: Contact Amruth
- **Missing values & duplicates**: Contact Tecla

---

## Quick Reference

### Running Tests
```bash
# Demo mode
python3 main.py

# With data
python3 main.py data.csv Target

# Test mock data
python3 mock_analysis_output.py

# Test suggestions
python3 preprocessing_suggestions.py
```

### Git Commands
```bash
# Check status
git status

# Add files
git add .

# Commit
git commit -m "Your message"

# Push
git push origin your-branch
```

---

**Last Updated**: October 2025

