#!/bin/bash

echo "🚀 Compiling InterviewMate Technical Report..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex not found. Please install LaTeX distribution."
    echo "   For macOS: brew install --cask mactex"
    echo "   For Ubuntu: sudo apt-get install texlive-full"
    exit 1
fi

# Create output directory
mkdir -p build

# First compilation
echo "📝 First compilation..."
pdflatex -output-directory=build technical_report.tex

# Second compilation for references
echo "📚 Second compilation (references)..."
pdflatex -output-directory=build technical_report.tex

# Third compilation for final formatting
echo "✨ Final compilation..."
pdflatex -output-directory=build technical_report.tex

# Check if PDF was created
if [ -f "build/technical_report.pdf" ]; then
    echo "✅ Success! PDF created: build/technical_report.pdf"
    echo "📊 File size: $(du -h build/technical_report.pdf | cut -f1)"
    
    # Open PDF if possible
    if command -v open &> /dev/null; then
        echo "🔍 Opening PDF..."
        open build/technical_report.pdf
    elif command -v xdg-open &> /dev/null; then
        echo "🔍 Opening PDF..."
        xdg-open build/technical_report.pdf
    fi
else
    echo "❌ Error: PDF compilation failed. Check build/technical_report.log for details."
    exit 1
fi

echo "🎉 Technical report compilation complete!"

