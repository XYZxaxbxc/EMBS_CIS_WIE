"use client"

import type React from "react"
import { useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Heart, Brain, Apple } from "lucide-react"
import { useAuth } from "@/lib/auth-context"

export default function LoginPage() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const { login, loading } = useAuth()
  const router = useRouter()

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    if (!email || !password) {
      setError("Veuillez remplir tous les champs")
      return
    }

    try {
      await login(email, password)
      router.push("/home")
    } catch (err) {
      setError("Erreur de connexion. Veuillez réessayer.")
      console.error(err)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background py-10">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <div className="flex justify-center gap-3 mb-4">
            <div className="p-3 bg-primary/10 rounded-lg">
              <Heart className="w-6 h-6 text-primary" />
            </div>
            <div className="p-3 bg-accent/10 rounded-lg">
              <Brain className="w-6 h-6 text-accent" />
            </div>
            <div className="p-3 bg-accent-secondary/10 rounded-lg">
              <Apple className="w-6 h-6 text-accent-secondary" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-foreground">Etma'En</h1>
          <p className="text-muted-foreground mt-2">
            Santé mentale & Nutrition pour toute la famille
          </p>
        </div>

        <Card className="border-0 shadow-lg">
          <CardHeader>
            <CardTitle>Connexion</CardTitle>
            <CardDescription>Accédez à votre compte Etma'En</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleLogin} className="space-y-4">
              {error && (
                <div className="p-3 bg-destructive/10 text-destructive rounded-lg text-sm">
                  {error}
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="vous@exemple.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">Mot de passe</Label>
                <Input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
              <Button type="submit" className="w-full btn-primary" disabled={loading}>
                {loading ? "Connexion..." : "Se connecter"}
              </Button>
            </form>

            <div className="mt-6 pt-6 border-t border-border">
              <p className="text-sm text-muted-foreground text-center mb-4">
                Pas encore de compte ?
              </p>
              <Link href="/signup">
                <Button variant="outline" className="w-full bg-transparent">
                  Créer un compte
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
