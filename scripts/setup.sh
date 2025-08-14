#!/bin/bash

# Aura-Vision Project Setup Script
# This script sets up the development environment for the Aura-Vision project

set -e  # Exit on any error

echo "🚀 Setting up Aura-Vision Development Environment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_success "Python found: $PYTHON_VERSION"
    else
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        print_success "pip found"
    else
        print_error "pip is not installed. Please install pip."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating Python virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    venv/Scripts/activate
    print_success "Virtual environment activated"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Python dependencies installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating project directories..."
    
    # Python directories
    mkdir -p python/models
    mkdir -p python/data
    mkdir -p python/utils
    mkdir -p python/notebooks
    
    # Data directories
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p data/augmented
    
    # Model directories
    mkdir -p models/checkpoints
    mkdir -p models/exported
    
    # Documentation
    mkdir -p docs
    
    # Scripts
    mkdir -p scripts
    
    print_success "Project directories created"
}

# Setup Git hooks (if Git is available)
setup_git_hooks() {
    if command -v git &> /dev/null; then
        print_status "Setting up Git hooks..."
        
        # Create pre-commit hook
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for Aura-Vision

echo "Running pre-commit checks..."

# Check Python syntax
find python -name "*.py" -exec python -m py_compile {} \;
if [ $? -ne 0 ]; then
    echo "Python syntax errors found!"
    exit 1
fi

# Run basic tests (if available)
if [ -f "python/test_basic.py" ]; then
    python python/test_basic.py
fi

echo "Pre-commit checks passed!"
EOF
        
        chmod +x .git/hooks/pre-commit
        print_success "Git hooks configured"
    else
        print_warning "Git not found, skipping Git hooks setup"
    fi
}

# Create initial configuration files
create_config_files() {
    print_status "Creating configuration files..."
    
    # Create .env file
    cat > .env << 'EOF'
# Aura-Vision Environment Configuration

# Model Configuration
MODEL_DEVICE=cpu
MODEL_PRECISION=fp32
BATCH_SIZE=16
LEARNING_RATE=0.0001

# Data Configuration
AUDIO_SAMPLE_RATE=16000
VIDEO_FRAME_RATE=30
AUDIO_SEGMENT_LENGTH=3.0

# Training Configuration
NUM_EPOCHS=100
VALIDATION_SPLIT=0.2
EARLY_STOPPING_PATIENCE=10

# Export Configuration
CORE_ML_VERSION=1.0
MODEL_QUANTIZATION=true
EOF
    
    # Create .gitignore
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
data/raw/
data/processed/
data/augmented/
models/checkpoints/
models/exported/
*.pth
*.pt
*.mlmodel
*.mlpackage

# Logs
logs/
*.log

# Environment variables
.env.local
.env.production

# Jupyter
.ipynb_checkpoints/

# Testing
.coverage
htmlcov/
.pytest_cache/
EOF
    
    print_success "Configuration files created"
}

# Check system requirements
check_system_requirements() {
    print_status "Checking system requirements..."
    
    # Check available memory
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY_GB" -lt 8 ]; then
            print_warning "Less than 8GB RAM detected. Training may be slow."
        else
            print_success "Sufficient RAM detected: ${MEMORY_GB}GB"
        fi
    fi
    
    # Check available disk space
    DISK_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$DISK_SPACE" -lt 10 ]; then
        print_warning "Less than 10GB free space detected. Consider freeing up space."
    else
        print_success "Sufficient disk space: ${DISK_SPACE}GB"
    fi
    
    # Check for CUDA (optional)
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected - CUDA support available"
    else
        print_warning "No NVIDIA GPU detected - training will use CPU"
    fi
}

# Main setup function
main() {
    echo "Starting Aura-Vision setup..."
    
    check_python
    check_pip
    check_system_requirements
    create_directories
    create_config_files
    create_venv
    activate_venv
    install_python_deps
    setup_git_hooks
    
    echo ""
    echo "🎉 Aura-Vision setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Open the iOS project in Xcode: open ios/AuraVision.xcodeproj"
    echo "3. Start training: python python/train.py --help"
    echo "4. Read the documentation: docs/ARCHITECTURE.md"
    echo ""
    echo "Happy coding! 🚀"
}

# Run main function
main "$@"
