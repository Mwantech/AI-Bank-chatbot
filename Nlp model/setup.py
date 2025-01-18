import subprocess
import sys
import nltk

def install_spacy_model():
    print("Installing spaCy English model...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    print("spaCy model installed successfully!")

def download_nltk_data():
    print("Downloading required NLTK data...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("NLTK data downloaded successfully!")

def main():
    try:
        print("Setting up required dependencies...")
        install_spacy_model()
        download_nltk_data()
        print("\nSetup completed successfully! You can now run the chatbot.")
        
    except Exception as e:
        print(f"An error occurred during setup: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()