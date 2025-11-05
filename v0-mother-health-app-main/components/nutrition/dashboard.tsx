"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

const dailyNutritionData = [
  { day: "Lun", calories: 1800, protein: 65, carbs: 220, fat: 60 },
  { day: "Mar", calories: 1950, protein: 70, carbs: 240, fat: 65 },
  { day: "Mer", calories: 1750, protein: 60, carbs: 210, fat: 55 },
  { day: "Jeu", calories: 2100, protein: 75, carbs: 260, fat: 70 },
  { day: "Ven", calories: 1900, protein: 68, carbs: 230, fat: 62 },
  { day: "Sam", calories: 2200, protein: 80, carbs: 270, fat: 75 },
  { day: "Dim", calories: 2000, protein: 72, carbs: 245, fat: 68 },
]

const weeklyGoals = [
  { label: "Calories", current: 1900, goal: 2000, unit: "kcal" },
  { label: "Protéines", current: 68, goal: 70, unit: "g" },
  { label: "Fruits & Légumes", current: 4, goal: 5, unit: "portions" },
  { label: "Eau", current: 1.8, goal: 2, unit: "L" },
]

export function NutritionDashboard() {
  const [selectedDay, setSelectedDay] = useState("Lun")

  return (
    <div className="space-y-6">
      {/* <CHANGE> Enhanced Goals Overview with modern gradients and hover effects */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {weeklyGoals.map((goal) => {
          const percentage = (goal.current / goal.goal) * 100
          const colorMap: { [key: string]: { bg: string; text: string; bar: string; borderColor: string; icon: string } } = {
            Calories: {
              bg: "bg-gradient-to-br from-purple-50 via-purple-100 to-indigo-150",
              text: "text-purple-700",
              bar: "bg-gradient-to-r from-purple-500 to-indigo-500",
              borderColor: "border-l-4 border-purple-500",
              icon: "🔥",
            },
            Protéines: {
              bg: "bg-gradient-to-br from-rose-50 via-pink-100 to-red-150",
              text: "text-rose-700",
              bar: "bg-gradient-to-r from-pink-500 to-rose-500",
              borderColor: "border-l-4 border-pink-500",
              icon: "💪",
            },
            "Fruits & Légumes": {
              bg: "bg-gradient-to-br from-emerald-50 via-green-100 to-teal-150",
              text: "text-emerald-700",
              bar: "bg-gradient-to-r from-green-500 to-emerald-500",
              borderColor: "border-l-4 border-green-500",
              icon: "🥗",
            },
            Eau: {
              bg: "bg-gradient-to-br from-sky-50 via-cyan-100 to-blue-150",
              text: "text-sky-700",
              bar: "bg-gradient-to-r from-cyan-500 to-blue-500",
              borderColor: "border-l-4 border-cyan-500",
              icon: "💧",
            },
          }
          const colors = colorMap[goal.label] || { bg: "bg-muted", text: "text-foreground", bar: "bg-primary", borderColor: "", icon: "" }

          return (
            <Card
              key={goal.label}
              className={`border-0 shadow-lg hover:shadow-2xl transition-all duration-300 ${colors.bg} ${colors.borderColor} hover:scale-105`}
            >
              <CardContent className="pt-6">
                <div className="space-y-3">
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-2">
                      <span className="text-2xl">{colors.icon}</span>
                      <h3 className={`font-bold text-foreground text-sm ${colors.text}`}>{goal.label}</h3>
                    </div>
                    <span className={`text-xs font-bold px-2 py-1 rounded-full ${colors.text} bg-white/50`}>
                      {Math.round(percentage)}%
                    </span>
                  </div>
                  <div className="w-full bg-white/60 rounded-full h-3 overflow-hidden shadow-inner">
                    <div
                      className={`h-3 rounded-full transition-all duration-500 ${colors.bar} shadow-md`}
                      style={{ width: `${Math.min(percentage, 100)}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground font-medium">
                    <span>
                      {goal.current} {goal.unit}
                    </span>
                    <span>
                      {goal.goal} {goal.unit}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* <CHANGE> Enhanced Calories Chart with gradient bars */}
      <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
        <CardHeader className="bg-gradient-to-r from-slate-50 to-slate-100">
          <CardTitle className="text-foreground">Calories par Jour</CardTitle>
          <CardDescription>Suivi hebdomadaire de votre consommation calorique</CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="h-[300px] flex items-end justify-around gap-3 p-6 bg-gradient-to-br from-slate-50 via-slate-100 to-slate-150 rounded-lg">
            {dailyNutritionData.map((data, idx) => (
              <div key={data.day} className="flex flex-col items-center gap-2 flex-1 group">
                <div
                  className="w-full rounded-t-lg transition-all duration-300 group-hover:shadow-lg group-hover:opacity-100 opacity-80 bg-gradient-to-t from-purple-600 via-purple-500 to-purple-400 shadow-md"
                  style={{ height: `${(data.calories / 2500) * 250}px` }}
                  title={`${data.calories} kcal`}
                />
                <span className="text-xs font-bold text-foreground group-hover:text-purple-600 transition-colors">{data.day}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* <CHANGE> Enhanced Macronutrients with improved styling */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
          <CardHeader className="bg-gradient-to-r from-violet-50 to-purple-50">
            <CardTitle>Distribution des Macronutriments</CardTitle>
            <CardDescription>Répartition moyenne de votre alimentation</CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="flex items-center justify-center gap-8">
              <div className="relative w-48 h-48 drop-shadow-lg">
                <svg viewBox="0 0 100 100" className="w-full h-full filter drop-shadow-md">
                  <circle cx="50" cy="50" r="45" fill="none" stroke="#f0f0f0" strokeWidth="30" />
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="url(#gradPurple)"
                    strokeWidth="30"
                    strokeDasharray={`${(25 / 100) * 283} 283`}
                    transform="rotate(-90 50 50)"
                    strokeLinecap="round"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="url(#gradPink)"
                    strokeWidth="30"
                    strokeDasharray={`${(50 / 100) * 283} 283`}
                    strokeDashoffset={`-${(25 / 100) * 283}`}
                    transform="rotate(-90 50 50)"
                    strokeLinecap="round"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="url(#gradCyan)"
                    strokeWidth="30"
                    strokeDasharray={`${(25 / 100) * 283} 283`}
                    strokeDashoffset={`-${((25 + 50) / 100) * 283}`}
                    transform="rotate(-90 50 50)"
                    strokeLinecap="round"
                  />
                  <defs>
                    <linearGradient id="gradPurple" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#a855f7" />
                      <stop offset="100%" stopColor="#7c3aed" />
                    </linearGradient>
                    <linearGradient id="gradPink" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#ec4899" />
                      <stop offset="100%" stopColor="#be185d" />
                    </linearGradient>
                    <linearGradient id="gradCyan" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#06b6d4" />
                      <stop offset="100%" stopColor="#0891b2" />
                    </linearGradient>
                  </defs>
                  <text x="50" y="45" textAnchor="middle" dy="0.3em" className="text-lg font-bold" fill="currentColor" fontSize="16">
                    100%
                  </text>
                </svg>
              </div>
              <div className="space-y-4">
                {[
                  { name: "Protéines", value: 25, color: "#a855f7", icon: "💪" },
                  { name: "Glucides", value: 50, color: "#ec4899", icon: "🌾" },
                  { name: "Lipides", value: 25, color: "#06b6d4", icon: "🥑" },
                ].map((macro) => (
                  <div key={macro.name} className="flex items-center gap-3 p-2 rounded-lg hover:bg-gray-50 transition-colors">
                    <span className="text-2xl">{macro.icon}</span>
                    <div className="flex-1">
                      <span className="text-sm font-semibold text-foreground">{macro.name}</span>
                      <div className="text-xs text-muted-foreground">{macro.value}%</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* <CHANGE> Enhanced Nutrition Trends with gradient bars */}
        <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
          <CardHeader className="bg-gradient-to-r from-cyan-50 to-blue-50">
            <CardTitle>Tendances Nutritionnelles</CardTitle>
            <CardDescription>Évolution de vos macronutriments</CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="space-y-5">
              {[
                { macro: "Protéines", value: 65, color: "from-purple-500 to-indigo-500", max: 100 },
                { macro: "Glucides", value: 220, color: "from-pink-500 to-rose-500", max: 300 },
                { macro: "Lipides", value: 60, color: "from-cyan-500 to-blue-500", max: 100 },
              ].map((item) => (
                <div key={item.macro}>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-bold text-foreground">{item.macro}</span>
                    <span className="text-sm font-semibold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                      {item.value}g
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden shadow-inner">
                    <div
                      className={`h-3 rounded-full transition-all duration-500 bg-gradient-to-r ${item.color} shadow-md`}
                      style={{
                        width: `${Math.min((item.value / item.max) * 100, 100)}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* <CHANGE> Enhanced Recommendations with better styling and icons */}
      <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
        <CardHeader className="bg-gradient-to-r from-amber-50 to-orange-50">
          <CardTitle>Recommandations Nutritionnelles</CardTitle>
          <CardDescription>Conseils personnalisés pour améliorer votre alimentation</CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="space-y-3">
            <div className="flex items-start gap-4 p-4 bg-gradient-to-r from-purple-50 to-violet-100 rounded-lg border-l-4 border-purple-500 hover:shadow-md transition-shadow">
              <span className="text-2xl font-bold text-purple-600 min-w-[40px]">1</span>
              <div className="flex-1">
                <p className="font-bold text-foreground text-sm">Augmentez votre consommation de fruits</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Vous êtes en dessous de votre objectif de 5 portions par jour
                </p>
              </div>
            </div>
            <div className="flex items-start gap-4 p-4 bg-gradient-to-r from-green-50 to-emerald-100 rounded-lg border-l-4 border-green-500 hover:shadow-md transition-shadow">
              <span className="text-2xl font-bold text-green-600 min-w-[40px]">2</span>
              <div className="flex-1">
                <p className="font-bold text-foreground text-sm">Excellent apport en protéines</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Vous maintenez un bon équilibre protéiné pour la croissance
                </p>
              </div>
            </div>
            <div className="flex items-start gap-4 p-4 bg-gradient-to-r from-cyan-50 to-sky-100 rounded-lg border-l-4 border-cyan-500 hover:shadow-md transition-shadow">
              <span className="text-2xl font-bold text-cyan-600 min-w-[40px]">3</span>
              <div className="flex-1">
                <p className="font-bold text-foreground text-sm">Hydratation à améliorer</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Essayez de boire plus d'eau tout au long de la journée
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}