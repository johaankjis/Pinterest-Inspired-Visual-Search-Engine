"use client"

import { Card } from "@/components/ui/card"
import { Sparkles } from "lucide-react"

interface SearchResult {
  id: string
  url: string
  similarity: number
  title: string
}

interface SearchResultsProps {
  results: SearchResult[]
}

export default function SearchResults({ results }: SearchResultsProps) {
  if (results.length === 0) {
    return (
      <div className="text-center py-16">
        <div className="w-16 h-16 rounded-full bg-muted mx-auto mb-4 flex items-center justify-center">
          <Sparkles className="w-8 h-8 text-muted-foreground" />
        </div>
        <h3 className="text-xl font-semibold text-foreground mb-2">No matches found</h3>
        <p className="text-muted-foreground">Try uploading a different image</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold text-foreground">Similar Images</h3>
          <p className="text-muted-foreground">Found {results.length} visually similar matches</p>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {results.map((result, index) => (
          <Card
            key={result.id}
            className="group overflow-hidden border-border/50 hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:scale-[1.02]"
          >
            <div className="relative aspect-square">
              <img src={result.url || "/placeholder.svg"} alt={result.title} className="w-full h-full object-cover" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <div className="absolute bottom-0 left-0 right-0 p-3 translate-y-full group-hover:translate-y-0 transition-transform duration-300">
                <div className="flex items-center justify-between text-white">
                  <span className="text-xs font-medium">Match #{index + 1}</span>
                  <span className="text-xs font-semibold bg-white/20 backdrop-blur-sm px-2 py-1 rounded">
                    {(result.similarity * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}
