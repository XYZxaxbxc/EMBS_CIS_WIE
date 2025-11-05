"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Brain, Apple, MessageSquare, TrendingUp } from "lucide-react"
import Link from "next/link"

export default function HomePage() {
  const services = [
    {
      icon: Brain,
      title: "Santé Mentale",
      description: "Chatbot IA et analyse des émotions par dessin",
      href: "/mental-health",
      color: "bg-accent/10",
      iconColor: "text-accent",
    },
    {
      icon: Apple,
      title: "Nutrition",
      description: "Suivi alimentaire et analyse nutritionnelle",
      href: "/nutrition",
      color: "bg-accent-secondary/10",
      iconColor: "text-accent-secondary",
    },
    {
      icon: MessageSquare,
      title: "Messagerie",
      description: "Communiquez avec votre médecin",
      href: "/messaging",
      color: "bg-primary/10",
      iconColor: "text-primary",
    },
    {
      icon: TrendingUp,
      title: "Rapports",
      description: "Suivez la progression et les analyses",
      href: "/reports",
      color: "bg-primary/10",
      iconColor: "text-primary",
    },
  ]

  return (
    <div className="p-6 space-y-8">
      {/* Welcome Section */}
      <div className="gradient-primary rounded-xl p-8 border border-primary/20">
        <h1 className="text-4xl font-bold text-foreground mb-2">Bienvenue sur Etma'en</h1>
        <p className="text-lg text-muted-foreground">
          Votre plateforme complète pour la santé mentale et nutritionnelle de votre famille
        </p>
      </div>

      {/* Services Grid */}
      <div>
        <h2 className="text-2xl font-bold text-foreground mb-6">Nos Services</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {services.map((service) => {
            const Icon = service.icon
            return (
              <Link key={service.href} href={service.href}>
                <Card className="card-hover cursor-pointer h-full">
                  <CardHeader>
                    <div className={`w-12 h-12 rounded-lg ${service.color} flex items-center justify-center mb-4`}>
                      <Icon className={`w-6 h-6 ${service.iconColor}`} />
                    </div>
                    <CardTitle className="text-lg">{service.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">{service.description}</p>
                  </CardContent>
                </Card>
              </Link>
            )
          })}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Dernière visite</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">Aujourd'hui</p>
            <p className="text-xs text-muted-foreground mt-1">À 14:30</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Santé mentale</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-accent">Bon</p>
            <p className="text-xs text-muted-foreground mt-1">Dernière analyse</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Équilibre nutritionnel</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-accent-secondary">78%</p>
            <p className="text-xs text-muted-foreground mt-1">Cette semaine</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
