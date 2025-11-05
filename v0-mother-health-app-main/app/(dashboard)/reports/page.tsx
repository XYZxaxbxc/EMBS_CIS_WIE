"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download } from "lucide-react"
import { Button } from "@/components/ui/button"

const monthlyData = [
  { month: "Jan", mental: 72, nutrition: 68 },
  { month: "Fév", mental: 75, nutrition: 70 },
  { month: "Mar", mental: 78, nutrition: 75 },
  { month: "Avr", mental: 82, nutrition: 78 },
  { month: "Mai", mental: 85, nutrition: 82 },
  { month: "Juin", mental: 88, nutrition: 85 },
]

export default function ReportsPage() {
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Rapports</h1>
          <p className="text-muted-foreground mt-2">Suivez votre progression au fil du temps</p>
        </div>
        <Button className="btn-primary gap-2">
          <Download className="w-4 h-4" />
          Exporter
        </Button>
      </div>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Vue d'ensemble</TabsTrigger>
          <TabsTrigger value="mental">Santé Mentale</TabsTrigger>
          <TabsTrigger value="nutrition">Nutrition</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Progression Globale</CardTitle>
              <CardDescription>Comparaison de votre santé mentale et nutritionnelle</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[300px] flex items-end justify-around gap-4 p-4 bg-muted/30 rounded-lg">
                {monthlyData.map((data) => (
                  <div key={data.month} className="flex flex-col items-center gap-2 flex-1">
                    <div className="flex gap-1 items-end h-full">
                      <div
                        className="bg-primary rounded-t transition-all hover:opacity-80"
                        style={{ width: "20px", height: `${(data.mental / 100) * 250}px` }}
                        title={`Santé Mentale: ${data.mental}`}
                      />
                      <div
                        className="bg-accent-secondary rounded-t transition-all hover:opacity-80"
                        style={{ width: "20px", height: `${(data.nutrition / 100) * 250}px` }}
                        title={`Nutrition: ${data.nutrition}`}
                      />
                    </div>
                    <span className="text-xs font-medium text-foreground">{data.month}</span>
                  </div>
                ))}
              </div>
              <div className="flex gap-4 justify-center mt-4">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-primary rounded" />
                  <span className="text-sm text-muted-foreground">Santé Mentale</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-accent-secondary rounded" />
                  <span className="text-sm text-muted-foreground">Nutrition</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="mental" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Santé Mentale - Détails</CardTitle>
              <CardDescription>Votre bien-être émotionnel au cours des 6 derniers mois</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground mb-2">Score actuel</p>
                  <p className="text-3xl font-bold text-accent">88/100</p>
                </div>
                <div className="space-y-2">
                  <p className="font-semibold text-foreground">Observations</p>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>✓ Amélioration constante depuis janvier</li>
                    <li>✓ Réduction du stress observée</li>
                    <li>✓ Meilleure qualité de sommeil</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="nutrition" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Nutrition - Détails</CardTitle>
              <CardDescription>Votre équilibre alimentaire au cours des 6 derniers mois</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground mb-2">Score actuel</p>
                  <p className="text-3xl font-bold text-accent-secondary">85/100</p>
                </div>
                <div className="space-y-2">
                  <p className="font-semibold text-foreground">Observations</p>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>✓ Équilibre macronutriments amélioré</li>
                    <li>✓ Consommation de fruits augmentée</li>
                    <li>✓ Réduction des sucres raffinés</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
