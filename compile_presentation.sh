#!/bin/bash

echo "🎯 Compiling InterviewMate Presentation..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex not found. Please install LaTeX distribution."
    echo "   For macOS: brew install --cask mactex"
    echo "   For Ubuntu: sudo apt-get install texlive-full"
    echo "   Alternative: Use online LaTeX compiler (Overleaf.com)"
    exit 1
fi

# Create output directory
mkdir -p build

# First compilation
echo "📝 First compilation..."
pdflatex -output-directory=build InterviewMate_Presentation.tex

# Second compilation for references
echo "📚 Second compilation (references)..."
pdflatex -output-directory=build InterviewMate_Presentation.tex

# Check if PDF was created
if [ -f "build/InterviewMate_Presentation.pdf" ]; then
    echo "✅ Success! Presentation PDF created: build/InterviewMate_Presentation.pdf"
    echo "📊 File size: $(du -h build/InterviewMate_Presentation.pdf | cut -f1)"
    
    # Open PDF if possible
    if command -v open &> /dev/null; then
        echo "🔍 Opening presentation..."
        open build/InterviewMate_Presentation.pdf
    elif command -v xdg-open &> /dev/null; then
        echo "🔍 Opening presentation..."
        xdg-open build/InterviewMate_Presentation.pdf
    fi
else
    echo "❌ Error: PDF compilation failed. Check build/InterviewMate_Presentation.log for details."
    echo "💡 Tip: Use online LaTeX compiler at Overleaf.com for instant results!"
    exit 1
fi

echo "🎉 Presentation compilation complete!"

