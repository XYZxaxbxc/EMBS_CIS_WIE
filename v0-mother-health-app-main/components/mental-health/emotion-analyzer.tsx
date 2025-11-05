"use client"

import type React from "react"
import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Upload, Loader, AlertCircle } from "lucide-react"

interface EmotionAnalysis {
  emotions: {
    name: string
    score: number
    color: string
  }[]
  summary: string
  recommendations: string[]
}

export function EmotionAnalyzer() {
  const [image, setImage] = useState<string | null>(null)
  const [analysis, setAnalysis] = useState<EmotionAnalysis | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Validate file type
    if (!file.type.startsWith("image/")) {
      setError("Veuillez sélectionner une image valide")
      return
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      setError("L'image doit faire moins de 5MB")
      return
    }

    const reader = new FileReader()
    reader.onload = (event) => {
      setImage(event.target?.result as string)
      setError("")
      analyzeDrawing(event.target?.result as string)
    }
    reader.readAsDataURL(file)
  }

  const analyzeDrawing = async (imageData: string) => {
    setLoading(true)
    setError("")

    try {
      // TODO: Integrate with AI SDK for real image analysis
      // For now, simulate analysis
      await new Promise((resolve) => setTimeout(resolve, 2000))

      const mockAnalysis: EmotionAnalysis = {
        emotions: [
          { name: "Joie", score: 0.85, color: "bg-yellow-500" },
          { name: "Sérénité", score: 0.72, color: "bg-blue-500" },
          { name: "Créativité", score: 0.68, color: "bg-purple-500" },
          { name: "Anxiété", score: 0.15, color: "bg-red-500" },
        ],
        summary:
          "Le dessin montre une expression créative positive avec des couleurs vives et des formes dynamiques. L'enfant semble être dans un état émotionnel stable et heureux.",
        recommendations: [
          "Encourager l'expression créative régulière",
          "Maintenir les activités ludiques actuelles",
          "Observer les changements dans les futurs dessins",
        ],
      }

      setAnalysis(mockAnalysis)
    } catch (err) {
      setError("Erreur lors de l'analyse. Veuillez réessayer.")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        className="border-2 border-dashed border-border rounded-lg p-8 text-center cursor-pointer hover:bg-muted/50 transition-colors"
        onClick={() => fileInputRef.current?.click()}
      >
        <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
        <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
        <p className="font-medium text-foreground mb-2">Uploadez un dessin</p>
        <p className="text-sm text-muted-foreground">Cliquez ou glissez-déposez une image (PNG, JPG, GIF)</p>
      </div>

      {error && (
        <div className="flex items-center gap-3 p-4 bg-destructive/10 text-destructive rounded-lg">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <p className="text-sm">{error}</p>
        </div>
      )}

      {/* Image Preview */}
      {image && (
        <div className="space-y-4">
          <div className="relative rounded-lg overflow-hidden border border-border">
            <img
              src={image || "/placeholder.svg"}
              alt="Uploaded drawing"
              className="w-full h-auto max-h-96 object-cover"
            />
          </div>

          {loading && (
            <div className="flex items-center justify-center p-8 bg-muted/30 rounded-lg">
              <div className="text-center">
                <Loader className="w-8 h-8 animate-spin mx-auto mb-2 text-primary" />
                <p className="text-sm text-muted-foreground">Analyse en cours...</p>
              </div>
            </div>
          )}

          {/* Analysis Results */}
          {analysis && !loading && (
            <div className="space-y-6">
              {/* Emotions Chart */}
              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold text-foreground mb-4">Émotions Détectées</h3>
                  <div className="space-y-3">
                    {analysis.emotions.map((emotion) => (
                      <div key={emotion.name}>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-sm font-medium text-foreground">{emotion.name}</span>
                          <span className="text-sm text-muted-foreground">{Math.round(emotion.score * 100)}%</span>
                        </div>
                        <div className="w-full bg-muted rounded-full h-2">
                          <div
                            className={`${emotion.color} h-2 rounded-full transition-all`}
                            style={{ width: `${emotion.score * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Summary */}
              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold text-foreground mb-3">Résumé</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">{analysis.summary}</p>
                </CardContent>
              </Card>

              {/* Recommendations */}
              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold text-foreground mb-3">Recommandations</h3>
                  <ul className="space-y-2">
                    {analysis.recommendations.map((rec, idx) => (
                      <li key={idx} className="flex items-start gap-3">
                        <span className="text-primary font-bold text-sm mt-0.5">{idx + 1}.</span>
                        <span className="text-sm text-muted-foreground">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>

              {/* New Analysis Button */}
              <Button
                onClick={() => {
                  setImage(null)
                  setAnalysis(null)
                  fileInputRef.current?.click()
                }}
                className="w-full btn-secondary"
              >
                Analyser un autre dessin
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
