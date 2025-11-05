"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"

export default function SettingsPage() {
  return (
    <div className="p-6 space-y-6 max-w-2xl">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Paramètres</h1>
        <p className="text-muted-foreground mt-2">Gérez vos préférences et paramètres de compte</p>
      </div>

      {/* Profile Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Profil</CardTitle>
          <CardDescription>Informations de votre compte médecin</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">Nom Complet</Label>
            <Input id="name" placeholder="Dr. Marie Dupont" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="specialty">Spécialité</Label>
            <Input id="specialty" placeholder="Pédiatre" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="license">Numéro de Licence</Label>
            <Input id="license" placeholder="XXXXX" />
          </div>
          <Button className="btn-primary">Enregistrer les modifications</Button>
        </CardContent>
      </Card>

      {/* Notification Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Notifications</CardTitle>
          <CardDescription>Gérez vos préférences de notification</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-foreground">Nouveaux messages</p>
              <p className="text-sm text-muted-foreground">Recevoir des alertes pour les nouveaux messages</p>
            </div>
            <Switch defaultChecked />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-foreground">Alertes patients</p>
              <p className="text-sm text-muted-foreground">Recevoir des alertes pour les changements de santé</p>
            </div>
            <Switch defaultChecked />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-foreground">Rapports hebdomadaires</p>
              <p className="text-sm text-muted-foreground">Recevoir un résumé hebdomadaire</p>
            </div>
            <Switch />
          </div>
        </CardContent>
      </Card>

      {/* Privacy Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Confidentialité</CardTitle>
          <CardDescription>Gérez vos paramètres de confidentialité</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-foreground">Profil public</p>
              <p className="text-sm text-muted-foreground">Permettre aux patients de voir votre profil</p>
            </div>
            <Switch defaultChecked />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-foreground">Partage de données</p>
              <p className="text-sm text-muted-foreground">Permettre le partage anonyme de données</p>
            </div>
            <Switch />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
