import { type NextRequest, NextResponse } from "next/server"

// Mock dataset of images with precomputed features
// In production, this would come from a database with actual feature vectors
const MOCK_DATASET = [
  {
    id: "1",
    url: "/mountain-sunset-vista.png",
    title: "Mountain Sunset",
    features: [0.8, 0.6, 0.3, 0.9, 0.4],
  },
  {
    id: "2",
    url: "/ocean-waves-beach.png",
    title: "Ocean Waves",
    features: [0.7, 0.5, 0.8, 0.3, 0.6],
  },
  {
    id: "3",
    url: "/forest-trees-nature.png",
    title: "Forest Path",
    features: [0.6, 0.9, 0.4, 0.7, 0.5],
  },
  {
    id: "4",
    url: "/city-skyline-night.png",
    title: "City Skyline",
    features: [0.4, 0.3, 0.9, 0.6, 0.8],
  },
  {
    id: "5",
    url: "/desert-sand-dunes.png",
    title: "Desert Dunes",
    features: [0.9, 0.4, 0.5, 0.8, 0.3],
  },
  {
    id: "6",
    url: "/waterfall-tropical.jpg",
    title: "Tropical Waterfall",
    features: [0.5, 0.8, 0.6, 0.4, 0.9],
  },
  {
    id: "7",
    url: "/snowy-mountain-peak.png",
    title: "Snowy Peak",
    features: [0.8, 0.7, 0.3, 0.9, 0.5],
  },
  {
    id: "8",
    url: "/autumn-leaves-colorful.jpg",
    title: "Autumn Colors",
    features: [0.6, 0.5, 0.7, 0.8, 0.4],
  },
]

// Simulate feature extraction from uploaded image
function extractFeatures(imageData: string): number[] {
  // In production, this would call the Python script with OpenCV
  // For demo purposes, generate random features
  return Array.from({ length: 5 }, () => Math.random())
}

// Calculate Euclidean distance between two feature vectors
function calculateDistance(features1: number[], features2: number[]): number {
  return Math.sqrt(features1.reduce((sum, val, i) => sum + Math.pow(val - features2[i], 2), 0))
}

// KNN-based similarity search
function findSimilarImages(queryFeatures: number[], k = 8) {
  // Calculate distances for all images in dataset
  const distances = MOCK_DATASET.map((item) => ({
    ...item,
    distance: calculateDistance(queryFeatures, item.features),
  }))

  // Sort by distance (ascending) and take top k
  distances.sort((a, b) => a.distance - b.distance)
  const topK = distances.slice(0, k)

  // Convert distance to similarity score (0-1)
  const maxDistance = Math.max(...topK.map((d) => d.distance))
  return topK.map((item) => ({
    id: item.id,
    url: item.url,
    title: item.title,
    similarity: 1 - item.distance / (maxDistance || 1),
  }))
}

export async function POST(request: NextRequest) {
  try {
    const { image } = await request.json()

    if (!image) {
      return NextResponse.json({ error: "No image provided" }, { status: 400 })
    }

    // Simulate processing delay (in production, this would be actual OpenCV processing)
    await new Promise((resolve) => setTimeout(resolve, 800))

    // Extract features from uploaded image
    const queryFeatures = extractFeatures(image)

    // Find similar images using KNN
    const results = findSimilarImages(queryFeatures)

    return NextResponse.json({ results })
  } catch (error) {
    console.error("[v0] Search error:", error)
    return NextResponse.json({ error: "Failed to process image" }, { status: 500 })
  }
}
