#!/bin/bash

# 1. Define variables
PROJECT_NAME="ClassificadorSinais"
JAR_NAME="${PROJECT_NAME}.jar"
OUT_DIR="bin"

echo "🧹 Cleaning up old builds..."
rm -rf $OUT_DIR
rm -f $JAR_NAME

echo "📁 Creating output directory..."
mkdir -p $OUT_DIR

echo "☕ Compiling Java source files..."
# Find all .java files in the current directory and subdirectories, then compile them into bin/
find . -name "*.java" > sources.txt
javac -d $OUT_DIR @sources.txt

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "📦 Packaging into executable JAR..."
    # c: create, f: file, e: entry point (Main class)
    # -C bin/ . : change to bin directory and include all files
    jar cfe $JAR_NAME Main -C $OUT_DIR .
    
    echo "✅ Build successful! Executable created: $JAR_NAME"
    
    # Clean up the temporary sources list
    rm sources.txt
    
    echo ""
    echo "🚀 To run your application, use the command:"
    echo "java -jar $JAR_NAME"
else
    echo "❌ Compilation failed. Check the errors above."
    rm sources.txt
    exit 1
fi