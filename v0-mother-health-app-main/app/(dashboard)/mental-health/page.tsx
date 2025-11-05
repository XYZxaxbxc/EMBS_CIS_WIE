"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { MessageCircle, Upload } from "lucide-react"
import { MentalHealthChatbot } from "@/components/mental-health/chatbot"
import { EmotionAnalyzer } from "@/components/mental-health/emotion-analyzer"

export default function MentalHealthPage() {
  const [activeTab, setActiveTab] = useState("chatbot")

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Santé Mentale</h1>
        <p className="text-muted-foreground mt-2">Explorez votre bien-être mental avec nos outils IA avancés</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="chatbot" className="flex items-center gap-2">
            <MessageCircle className="w-4 h-4" />
            Chatbot IA
          </TabsTrigger>
          <TabsTrigger value="emotion" className="flex items-center gap-2">
            <Upload className="w-4 h-4" />
            Analyse d'Émotions
          </TabsTrigger>
        </TabsList>

        <TabsContent value="chatbot" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Assistant Santé Mentale</CardTitle>
              <CardDescription>Discutez de votre bien-être mental ou de celui de votre enfant</CardDescription>
            </CardHeader>
            <CardContent>
              <MentalHealthChatbot />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="emotion" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Analyse des Émotions par Dessin</CardTitle>
              <CardDescription>Uploadez un dessin de votre enfant pour analyser ses émotions</CardDescription>
            </CardHeader>
            <CardContent>
              <EmotionAnalyzer />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
