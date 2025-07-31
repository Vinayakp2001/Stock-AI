#!/usr/bin/env python3
"""
Setup script for Stock Prediction Agent SDK
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "data",
        "logs",
        "reports"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def test_installation():
    """Test if the installation works"""
    print("🧪 Testing installation...")
    try:
        # Test imports
        from agents.data_agent import DataAgent
        from agents.prediction_agent import PredictionAgent
        from backtesting.engine import BacktestEngine
        
        print("✅ All modules imported successfully!")
        
        # Test data fetching
        data_agent = DataAgent()
        stock_data = data_agent.get_stock_data('AAPL', period='1mo')
        print(f"✅ Data fetching works! Got {len(stock_data.data)} data points for AAPL")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Stock Prediction Agent SDK - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Test installation
    print("\n🧪 Testing installation...")
    if not test_installation():
        print("❌ Installation test failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📖 Next steps:")
    print("1. Run the demo: python main.py --demo")
    print("2. Analyze a stock: python main.py --symbol AAPL")
    print("3. Run backtesting: python main.py --symbol AAPL --backtest")
    print("4. Start web dashboard: python dashboard/app.py")
    print("\n⚠️  Remember: This is for educational purposes only!")
    print("   Always do your own research before making investment decisions.")

if __name__ == "__main__":
    main() 