"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Send, Loader } from "lucide-react"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

export function NutritionChatbot() {
  // Use NEXT_PUBLIC_NUTRITION_BACKEND_URL to allow changing backend address in dev/prod.
  // Default to the nutritionist backend port used in this repo (3000).
  const BACKEND_URL = (process.env.NEXT_PUBLIC_NUTRITION_BACKEND_URL as string) || "http://127.0.0.1:9000"
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content:
        "Bonjour! 👋 Je suis votre assistant nutritionniste. Parlez-moi de votre alimentation ou de vos objectifs : perte de poids, recettes équilibrées, ou régimes adaptés.",
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setLoading(true)

    try {
      const url = `${BACKEND_URL.replace(/\/$/, "")}/api/nutrition/recipes`
      console.log("Sending nutrition request to:", url)
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: input,
          conversationHistory: messages,
        }),
      })

      if (!response.ok) {
        const text = await response.text()
        throw new Error(`Erreur serveur: ${response.status} - ${text}`)
      }

      const data = await response.json()
      console.log("Réponse du backend:", data)

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response || "Je peux vous proposer un plan nutritionnel équilibré si vous me dites vos préférences.",
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error("Erreur:", error)
      const fallbackMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "Je n’ai pas pu contacter le serveur pour le moment. Mais je peux vous aider à créer un régime de base : voulez-vous perdre du poids, prendre de la masse ou manger plus équilibré ?",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, fallbackMessage])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-[600px] bg-muted/30 rounded-lg border border-border">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                message.role === "user"
                  ? "bg-primary text-primary-foreground rounded-br-none"
                  : "bg-card border border-border rounded-bl-none"
              }`}
            >
              <p className="text-sm whitespace-pre-line">{message.content}</p>
              <span className="text-xs opacity-70 mt-1 block">
                {message.timestamp.toLocaleTimeString("fr-FR", {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-card border border-border px-4 py-2 rounded-lg rounded-bl-none">
              <Loader className="w-4 h-4 animate-spin" />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-border p-4">
        <form onSubmit={handleSendMessage} className="flex gap-2">
          <Input
            placeholder="Décrivez votre alimentation ou vos objectifs..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
            className="flex-1"
          />
          <Button type="submit" disabled={loading || !input.trim()} className="btn-primary">
            <Send className="w-4 h-4" />
          </Button>
        </form>
      </div>
    </div>
  )
}
