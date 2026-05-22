#!/bin/bash

# 1. Define variables
PROJECT_NAME="ClassificadorSinais"
JAR_NAME="${PROJECT_NAME}.jar"
OUT_DIR="bin"
SRC_DIR="src"

echo "🧹 Cleaning up old builds..."
rm -rf $OUT_DIR
rm -f $JAR_NAME

echo "📁 Creating output directory..."
mkdir -p $OUT_DIR

echo "☕ Compiling Java source files..."
find $SRC_DIR -name "*.java" > sources.txt

javac -sourcepath $SRC_DIR -d $OUT_DIR @sources.txt

if [ $? -eq 0 ]; then
    echo "📦 Packaging into executable JAR..."
    jar cfe $JAR_NAME Main -C $OUT_DIR .
    
    echo "✅ Build successful! Executable created: $JAR_NAME"
    
    rm sources.txt
    echo ""
    echo "🚀 To run your application, use the command:"
    echo "java -jar $JAR_NAME"
else
    echo "❌ Compilation failed. Check the errors above."
    rm sources.txt
    exit 1
fi