"use client"

import { useState } from "react"
import { Upload, Search, Sparkles, ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import ImageUpload from "@/components/image-upload"
import SearchResults from "@/components/search-results"

export default function VisualSearchPage() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [isSearching, setIsSearching] = useState(false)
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [hasSearched, setHasSearched] = useState(false)

  const handleImageUpload = async (imageData: string) => {
    setUploadedImage(imageData)
    setIsSearching(true)
    setHasSearched(false)

    try {
      // Call the search API
      const response = await fetch("/api/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: imageData }),
      })

      const data = await response.json()
      setSearchResults(data.results || [])
      setHasSearched(true)
    } catch (error) {
      console.error("[v0] Error searching:", error)
      setSearchResults([])
      setHasSearched(true)
    } finally {
      setIsSearching(false)
    }
  }

  const handleReset = () => {
    setUploadedImage(null)
    setSearchResults([])
    setHasSearched(false)
    setIsSearching(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      {/* Header */}
      <header className="border-b border-border/40 bg-background/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-chart-1 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">VisualFind</h1>
              <p className="text-xs text-muted-foreground">AI-Powered Image Search</p>
            </div>
          </div>
          <Button variant="outline" size="sm" onClick={handleReset}>
            <Search className="w-4 h-4 mr-2" />
            New Search
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-12">
        {!uploadedImage ? (
          <div className="max-w-4xl mx-auto">
            {/* Hero Section */}
            <div className="text-center mb-12 space-y-4">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-4">
                <Sparkles className="w-4 h-4" />
                Powered by Computer Vision & ML
              </div>
              <h2 className="text-5xl font-bold text-balance text-foreground">Discover Visually Similar Images</h2>
              <p className="text-xl text-muted-foreground text-balance max-w-2xl mx-auto">
                Upload any image and let our AI find visually similar matches using advanced feature extraction and KNN
                algorithms
              </p>
            </div>

            {/* Upload Section */}
            <ImageUpload onImageUpload={handleImageUpload} />

            {/* Features */}
            <div className="grid md:grid-cols-3 gap-6 mt-16">
              <Card className="p-6 space-y-3 border-border/50 bg-card/50 backdrop-blur-sm">
                <div className="w-12 h-12 rounded-lg bg-chart-1/20 flex items-center justify-center">
                  <Upload className="w-6 h-6 text-chart-1" />
                </div>
                <h3 className="font-semibold text-foreground">Fast Upload</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Drag & drop or click to upload. Processing completes in under 200ms with optimized preprocessing.
                </p>
              </Card>

              <Card className="p-6 space-y-3 border-border/50 bg-card/50 backdrop-blur-sm">
                <div className="w-12 h-12 rounded-lg bg-chart-2/20 flex items-center justify-center">
                  <Sparkles className="w-6 h-6 text-chart-2" />
                </div>
                <h3 className="font-semibold text-foreground">AI-Powered Matching</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Advanced feature extraction using OpenCV with 85%+ accuracy on visual similarity scoring.
                </p>
              </Card>

              <Card className="p-6 space-y-3 border-border/50 bg-card/50 backdrop-blur-sm">
                <div className="w-12 h-12 rounded-lg bg-chart-3/20 flex items-center justify-center">
                  <Search className="w-6 h-6 text-chart-3" />
                </div>
                <h3 className="font-semibold text-foreground">Instant Results</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Get top matches in under 1 second with optimized KNN retrieval and indexed data structures.
                </p>
              </Card>
            </div>
          </div>
        ) : (
          <div className="space-y-8">
            {/* Query Image */}
            <div className="max-w-2xl mx-auto">
              <div className="flex items-center gap-3 mb-4">
                <ImageIcon className="w-5 h-5 text-muted-foreground" />
                <h3 className="text-lg font-semibold text-foreground">Your Search Image</h3>
              </div>
              <Card className="overflow-hidden border-border/50">
                <img
                  src={uploadedImage || "/placeholder.svg"}
                  alt="Uploaded query"
                  className="w-full h-auto max-h-96 object-contain bg-muted"
                />
              </Card>
            </div>

            {/* Results */}
            {isSearching ? (
              <div className="text-center py-16">
                <div className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-primary/10">
                  <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                  <span className="text-sm font-medium text-primary">Analyzing image features...</span>
                </div>
              </div>
            ) : hasSearched ? (
              <SearchResults results={searchResults} />
            ) : null}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border/40 mt-20 py-8">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>Built with OpenCV, Scikit-learn KNN, and Next.js • Optimized for speed and accuracy</p>
        </div>
      </footer>
    </div>
  )
}
