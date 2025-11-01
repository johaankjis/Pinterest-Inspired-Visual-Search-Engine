"use client"

import type React from "react"

import { useCallback, useState } from "react"
import { Upload, ImageIcon } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

interface ImageUploadProps {
  onImageUpload: (imageData: string) => void
}

export default function ImageUpload({ onImageUpload }: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false)

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) {
        alert("Please upload an image file")
        return
      }

      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result as string
        onImageUpload(result)
      }
      reader.readAsDataURL(file)
    },
    [onImageUpload],
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)

      const file = e.dataTransfer.files[0]
      if (file) {
        handleFile(file)
      }
    },
    [handleFile],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        handleFile(file)
      }
    },
    [handleFile],
  )

  return (
    <Card
      className={`relative border-2 border-dashed transition-all duration-200 ${
        isDragging
          ? "border-primary bg-primary/5 scale-[1.02]"
          : "border-border/50 bg-card/50 hover:border-primary/50 hover:bg-card/80"
      }`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <div className="p-12 text-center space-y-6">
        <div className="flex justify-center">
          <div className="w-20 h-20 rounded-full bg-gradient-to-br from-primary/20 to-chart-1/20 flex items-center justify-center">
            {isDragging ? (
              <ImageIcon className="w-10 h-10 text-primary animate-pulse" />
            ) : (
              <Upload className="w-10 h-10 text-primary" />
            )}
          </div>
        </div>

        <div className="space-y-2">
          <h3 className="text-2xl font-semibold text-foreground">
            {isDragging ? "Drop your image here" : "Upload an image to search"}
          </h3>
          <p className="text-muted-foreground">Drag and drop or click to browse • Supports JPG, PNG, WebP</p>
        </div>

        <div className="flex justify-center">
          <Button size="lg" className="relative overflow-hidden group">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <Upload className="w-5 h-5 mr-2 group-hover:scale-110 transition-transform" />
            Choose Image
          </Button>
        </div>

        <p className="text-xs text-muted-foreground">Maximum file size: 10MB • Optimized for best results</p>
      </div>
    </Card>
  )
}
