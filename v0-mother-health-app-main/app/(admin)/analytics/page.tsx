"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

const mentalHealthData = [
  { range: "0-20", count: 0, color: "#ef4444" },
  { range: "20-40", count: 1, color: "#f97316" },
  { range: "40-60", count: 2, color: "#eab308" },
  { range: "60-80", count: 3, color: "#84cc16" },
  { range: "80-100", count: 2, color: "#10b981" },
]

const issueDistribution = [
  { name: "Anxiété", value: 35, color: "#3b82f6" },
  { name: "Stress", value: 25, color: "#8b5cf6" },
  { name: "Déséquilibre Nutritionnel", value: 20, color: "#f59e0b" },
  { name: "Autres", value: 20, color: "#6b7280" },
]

export default function AnalyticsPage() {
  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Analyses</h1>
        <p className="text-muted-foreground mt-2">Vue d'ensemble des données de santé de vos patients</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground mb-1">Santé Mentale Moyenne</p>
            <p className="text-3xl font-bold text-primary">75.8%</p>
            <p className="text-xs text-green-600 mt-2">+2.3% ce mois</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground mb-1">Nutrition Moyenne</p>
            <p className="text-3xl font-bold text-accent-secondary">72.5%</p>
            <p className="text-xs text-green-600 mt-2">+1.8% ce mois</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground mb-1">Taux de Suivi</p>
            <p className="text-3xl font-bold text-accent">92%</p>
            <p className="text-xs text-muted-foreground mt-2">Patients actifs</p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Mental Health Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Distribution Santé Mentale</CardTitle>
            <CardDescription>Répartition des scores de santé mentale</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] flex items-end justify-around gap-2 p-4 bg-muted/30 rounded-lg">
              {mentalHealthData.map((data) => (
                <div key={data.range} className="flex flex-col items-center gap-2 flex-1">
                  <div
                    className="w-full rounded-t transition-all hover:opacity-80"
                    style={{
                      height: `${data.count * 50}px`,
                      backgroundColor: data.color,
                    }}
                    title={`${data.range}: ${data.count} patients`}
                  />
                  <span className="text-xs font-medium text-foreground text-center">{data.range}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Issue Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Distribution des Problèmes</CardTitle>
            <CardDescription>Types de problèmes identifiés</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center gap-8">
              <div className="relative w-40 h-40">
                <svg viewBox="0 0 100 100" className="w-full h-full">
                  <circle cx="50" cy="50" r="40" fill="none" stroke="#e5e7eb" strokeWidth="25" />
                  {issueDistribution.map((offset, issue, idx) => {
                    const dasharray = `${(issue.value / 100) * 251} 251`
                    return (
                      <>
                        {offset}
                        <circle
                          key={issue.name}
                          cx="50"
                          cy="50"
                          r="40"
                          fill="none"
                          stroke={issue.color}
                          strokeWidth="25"
                          strokeDasharray={dasharray}
                          strokeDashoffset={`-${idx > 0 ? issueDistribution.slice(0, idx).reduce((sum, i) => sum + (i.value / 100) * 251, 0) : 0}`}
                          transform="rotate(-90 50 50)"/>
                      </>
                    )
                }, null)} 
                </svg>
              </div>
              <div className="space-y-3">
                {issueDistribution.map((issue) => (
                  <div key={issue.name} className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: issue.color }} />
                    <span className="text-sm text-foreground">{issue.name}</span>
                    <span className="text-sm font-semibold text-muted-foreground">{issue.value}%</span>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Nutrition Trend */}
      <Card>
        <CardHeader>
          <CardTitle>Tendance Nutritionnelle</CardTitle>
          <CardDescription>Évolution du score nutritionnel moyen</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] flex items-end justify-around gap-4 p-4 bg-muted/30 rounded-lg">
            {[
              { month: "Août", avg: 68 },
              { month: "Septembre", avg: 71 },
              { month: "Octobre", avg: 74 },
            ].map((data) => (
              <div key={data.month} className="flex flex-col items-center gap-2 flex-1">
                <div
                  className="w-full bg-accent-secondary rounded-t transition-all hover:opacity-80"
                  style={{ height: `${(data.avg / 100) * 250}px` }}
                  title={`${data.month}: ${data.avg}`}
                />
                <span className="text-xs font-medium text-foreground">{data.month}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
