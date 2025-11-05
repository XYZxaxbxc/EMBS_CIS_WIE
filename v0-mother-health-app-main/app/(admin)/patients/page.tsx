"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { XAxis, YAxis, CartesianGrid, ResponsiveContainer, LineChart, Line } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Search, Plus, Eye, MessageSquare } from "lucide-react"

interface Patient {
  id: string
  name: string
  age: number
  lastVisit: string
  status: "healthy" | "monitoring" | "concern"
  mentalHealth: number
  nutrition: number
}

const patients: Patient[] = [
  {
    id: "1",
    name: "Emma Dupont",
    age: 8,
    lastVisit: "2024-10-28",
    status: "healthy",
    mentalHealth: 85,
    nutrition: 78,
  },
  {
    id: "2",
    name: "Lucas Martin",
    age: 10,
    lastVisit: "2024-10-25",
    status: "monitoring",
    mentalHealth: 72,
    nutrition: 65,
  },
  {
    id: "3",
    name: "Sophie Bernard",
    age: 7,
    lastVisit: "2024-10-20",
    status: "concern",
    mentalHealth: 58,
    nutrition: 55,
  },
  {
    id: "4",
    name: "Thomas Lefevre",
    age: 9,
    lastVisit: "2024-10-28",
    status: "healthy",
    mentalHealth: 88,
    nutrition: 82,
  },
]

const healthTrendData = [
  { month: "Août", mentalHealth: 75, nutrition: 70 },
  { month: "Septembre", mentalHealth: 78, nutrition: 72 },
  { month: "Octobre", mentalHealth: 80, nutrition: 75 },
]

const statusDistribution = [
  { name: "Sain", value: 2, color: "#10b981" },
  { name: "Suivi", value: 1, color: "#f59e0b" },
  { name: "Préoccupant", value: 1, color: "#ef4444" },
]

export default function PatientsPage() {
  const [searchQuery, setSearchQuery] = useState("")

  const filteredPatients = patients.filter((p) => p.name.toLowerCase().includes(searchQuery.toLowerCase()))

  const getStatusColor = (status: string) => {
    switch (status) {
      case "healthy":
        return "bg-success/10 text-success"
      case "monitoring":
        return "bg-warning/10 text-warning"
      case "concern":
        return "bg-destructive/10 text-destructive"
      default:
        return "bg-muted text-muted-foreground"
    }
  }

  const getStatusLabel = (status: string) => {
    switch (status) {
      case "healthy":
        return "Sain"
      case "monitoring":
        return "Suivi"
      case "concern":
        return "Préoccupant"
      default:
        return status
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Mes Patients</h1>
          <p className="text-muted-foreground mt-2">Gérez et suivez la santé de vos patients</p>
        </div>
        <Button className="btn-primary">
          <Plus className="w-4 h-4 mr-2" />
          Ajouter un patient
        </Button>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground mb-1">Total Patients</p>
            <p className="text-3xl font-bold text-foreground">{patients.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground mb-1">Sains</p>
            <p className="text-3xl font-bold text-success">{patients.filter((p) => p.status === "healthy").length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground mb-1">Sous Suivi</p>
            <p className="text-3xl font-bold text-warning">
              {patients.filter((p) => p.status === "monitoring").length}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground mb-1">Préoccupants</p>
            <p className="text-3xl font-bold text-destructive">
              {patients.filter((p) => p.status === "concern").length}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Health Trends */}
      <Card>
        <CardHeader>
          <CardTitle>Tendances de Santé Globale</CardTitle>
          <CardDescription>Évolution moyenne de la santé mentale et nutritionnelle</CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer
            config={{
              mentalHealth: { label: "Santé Mentale", color: "hsl(var(--chart-1))" },
              nutrition: { label: "Nutrition", color: "hsl(var(--chart-2))" },
            }}
            className="h-[300px]"
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={healthTrendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Line type="monotone" dataKey="mentalHealth" stroke="var(--color-mentalHealth)" />
                <Line type="monotone" dataKey="nutrition" stroke="var(--color-nutrition)" />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* Patients List */}
      <Card>
        <CardHeader>
          <CardTitle>Liste des Patients</CardTitle>
          <div className="mt-4 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Rechercher un patient..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 px-4 font-medium text-muted-foreground">Nom</th>
                  <th className="text-left py-3 px-4 font-medium text-muted-foreground">Âge</th>
                  <th className="text-left py-3 px-4 font-medium text-muted-foreground">Dernière Visite</th>
                  <th className="text-left py-3 px-4 font-medium text-muted-foreground">Santé Mentale</th>
                  <th className="text-left py-3 px-4 font-medium text-muted-foreground">Nutrition</th>
                  <th className="text-left py-3 px-4 font-medium text-muted-foreground">Statut</th>
                  <th className="text-left py-3 px-4 font-medium text-muted-foreground">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredPatients.map((patient) => (
                  <tr key={patient.id} className="border-b border-border hover:bg-muted/50 transition-colors">
                    <td className="py-3 px-4 font-medium text-foreground">{patient.name}</td>
                    <td className="py-3 px-4 text-muted-foreground">{patient.age} ans</td>
                    <td className="py-3 px-4 text-muted-foreground">{patient.lastVisit}</td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-16 bg-muted rounded-full h-2">
                          <div className="bg-primary h-2 rounded-full" style={{ width: `${patient.mentalHealth}%` }} />
                        </div>
                        <span className="text-xs font-medium">{patient.mentalHealth}%</span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-16 bg-muted rounded-full h-2">
                          <div
                            className="bg-accent-secondary h-2 rounded-full"
                            style={{ width: `${patient.nutrition}%` }}
                          />
                        </div>
                        <span className="text-xs font-medium">{patient.nutrition}%</span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(patient.status)}`}>
                        {getStatusLabel(patient.status)}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          <Eye className="w-4 h-4" />
                        </Button>
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          <MessageSquare className="w-4 h-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
