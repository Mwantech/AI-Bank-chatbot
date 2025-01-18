import subprocess
import sys
import time

def install_dependencies():
    print("Setting up dependencies...")
    
    # First uninstall conflicting packages
    packages_to_uninstall = [
        "numpy",
        "spacy",
        "scikit-learn",
        "h5py"
    ]
    
    for package in packages_to_uninstall:
        print(f"Removing {package} if it exists...")
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", package])
        time.sleep(1)  # Give system time to clean up
    
    # Install packages in the correct order with specific versions
    packages_to_install = [
        "numpy==1.24.3",  # Using a stable version compatible with other packages
        "scikit-learn==1.3.0",
        "spacy==3.5.3",
        "nltk==3.8.1"
    ]
    
    for package in packages_to_install:
        print(f"\nInstalling {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {str(e)}")
            return False
        time.sleep(1)  # Give system time between installations
    
    # Download spaCy model
    print("\nDownloading spaCy English model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("Successfully downloaded spaCy model")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading spaCy model: {str(e)}")
        return False
    
    return True

def setup_nltk():
    print("\nDownloading NLTK data...")
    import nltk
    resources = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")
            return False
    return True

def main():
    print("Starting dependency setup...")
    
    if not install_dependencies():
        print("Failed to install dependencies")
        return
    
    if not setup_nltk():
        print("Failed to setup NLTK")
        return
    
    print("\nSetup completed successfully! You can now run the chatbot.")

if __name__ == "__main__":
    main()