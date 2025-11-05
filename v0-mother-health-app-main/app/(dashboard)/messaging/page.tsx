"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { MessageCircle, Send, Loader, Phone, Video, MoreVertical, Search } from "lucide-react"
import { useAuth } from "@/lib/auth-context"

interface Conversation {
  id: string
  doctorName: string
  doctorRole: string
  lastMessage: string
  timestamp: string
  unread: number
  avatar: string
  status: "online" | "offline"
}

interface Message {
  id: string
  sender: "user" | "doctor"
  senderName: string
  content: string
  timestamp: Date
  read: boolean
}

export default function MessagingPage() {
  const { user } = useAuth()
  const [selectedConversation, setSelectedConversation] = useState<string>("1")
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      sender: "doctor",
      senderName: "Dr. Marie Dupont",
      content: "Bonjour, comment allez-vous? Avez-vous des questions concernant la santé de votre enfant?",
      timestamp: new Date(Date.now() - 3600000),
      read: true,
    },
  ])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const conversations: Conversation[] = [
    {
      id: "1",
      doctorName: "Dr. Marie Dupont",
      doctorRole: "Pédiatre",
      lastMessage: "Comment allez-vous? Avez-vous des questions...",
      timestamp: "14:30",
      unread: 0,
      avatar: "MD",
      status: "online",
    },
    {
      id: "2",
      doctorName: "Dr. Jean Martin",
      doctorRole: "Nutritionniste",
      lastMessage: "Les résultats sont bons, continuez...",
      timestamp: "10:15",
      unread: 2,
      avatar: "JM",
      status: "offline",
    },
    {
      id: "3",
      doctorName: "Dr. Sophie Bernard",
      doctorRole: "Psychologue",
      lastMessage: "À bientôt pour notre prochaine session",
      timestamp: "Hier",
      unread: 0,
      avatar: "SB",
      status: "online",
    },
  ]

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
      sender: "user",
      senderName: user?.name || "Vous",
      content: input,
      timestamp: new Date(),
      read: false,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setLoading(true)

    try {
      const response = await fetch("/api/messaging/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversationId: selectedConversation,
          message: input,
          senderName: user?.name,
        }),
      })

      if (!response.ok) throw new Error("Failed to send message")

      // Simulate doctor response
      setTimeout(() => {
        const doctorMessage: Message = {
          id: (Date.now() + 1).toString(),
          sender: "doctor",
          senderName: "Dr. Marie Dupont",
          content: "Merci pour votre message. Je vais examiner cela et vous revenir rapidement.",
          timestamp: new Date(),
          read: false,
        }
        setMessages((prev) => [...prev, doctorMessage])
        setLoading(false)
      }, 1000)
    } catch (error) {
      console.error("Error sending message:", error)
      setLoading(false)
    }
  }

  const selectedConv = conversations.find((c) => c.id === selectedConversation)
  const filteredConversations = conversations.filter((conv) =>
    conv.doctorName.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Messagerie</h1>
        <p className="text-muted-foreground mt-2">Communiquez directement avec vos médecins</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-200px)]">
        {/* Conversations List */}
        <Card className="lg:col-span-1 flex flex-col overflow-hidden">
          <CardHeader className="border-b border-border pb-4">
            <CardTitle className="text-lg mb-4">Conversations</CardTitle>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Rechercher..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
          </CardHeader>
          <CardContent className="flex-1 overflow-y-auto p-0">
            <div className="space-y-1">
              {filteredConversations.map((conv) => (
                <button
                  key={conv.id}
                  onClick={() => setSelectedConversation(conv.id)}
                  className={`w-full text-left p-4 border-b border-border hover:bg-muted transition-colors ${
                    selectedConversation === conv.id ? "bg-muted" : ""
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className="relative">
                      <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center text-sm font-semibold text-primary">
                        {conv.avatar}
                      </div>
                      <div
                        className={`absolute bottom-0 right-0 w-3 h-3 rounded-full border-2 border-background ${
                          conv.status === "online" ? "bg-success" : "bg-muted-foreground"
                        }`}
                      />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <p className="font-medium text-foreground text-sm">{conv.doctorName}</p>
                        {conv.unread > 0 && (
                          <span className="bg-primary text-primary-foreground text-xs rounded-full w-5 h-5 flex items-center justify-center">
                            {conv.unread}
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground mb-1">{conv.doctorRole}</p>
                      <p className="text-xs text-muted-foreground truncate">{conv.lastMessage}</p>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground mt-2 ml-13">{conv.timestamp}</p>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Chat Area */}
        <Card className="lg:col-span-2 flex flex-col overflow-hidden">
          {selectedConv ? (
            <>
              {/* Header */}
              <CardHeader className="border-b border-border pb-4 flex flex-row items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center text-sm font-semibold text-primary">
                      {selectedConv.avatar}
                    </div>
                    <div
                      className={`absolute bottom-0 right-0 w-3 h-3 rounded-full border-2 border-background ${
                        selectedConv.status === "online" ? "bg-success" : "bg-muted-foreground"
                      }`}
                    />
                  </div>
                  <div>
                    <CardTitle className="text-base">{selectedConv.doctorName}</CardTitle>
                    <p className="text-xs text-muted-foreground">
                      {selectedConv.status === "online" ? "En ligne" : "Hors ligne"}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="ghost" size="icon" className="h-9 w-9">
                    <Phone className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="icon" className="h-9 w-9">
                    <Video className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="icon" className="h-9 w-9">
                    <MoreVertical className="w-4 h-4" />
                  </Button>
                </div>
              </CardHeader>

              {/* Messages */}
              <CardContent className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div className={`flex gap-2 max-w-xs ${message.sender === "user" ? "flex-row-reverse" : ""}`}>
                      {message.sender === "doctor" && (
                        <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-xs font-semibold text-primary flex-shrink-0">
                          {selectedConv.avatar}
                        </div>
                      )}
                      <div>
                        <div
                          className={`px-4 py-2 rounded-lg ${
                            message.sender === "user"
                              ? "bg-primary text-primary-foreground rounded-br-none"
                              : "bg-muted rounded-bl-none"
                          }`}
                        >
                          <p className="text-sm">{message.content}</p>
                        </div>
                        <span className="text-xs text-muted-foreground mt-1 block">
                          {message.timestamp.toLocaleTimeString("fr-FR", {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex justify-start">
                    <div className="flex gap-2">
                      <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-xs font-semibold text-primary flex-shrink-0">
                        {selectedConv.avatar}
                      </div>
                      <div className="bg-muted px-4 py-2 rounded-lg rounded-bl-none">
                        <Loader className="w-4 h-4 animate-spin" />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </CardContent>

              {/* Input */}
              <div className="border-t border-border p-4">
                <form onSubmit={handleSendMessage} className="flex gap-2">
                  <Input
                    placeholder="Écrivez votre message..."
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
            </>
          ) : (
            <CardContent className="flex items-center justify-center h-full">
              <div className="text-center">
                <MessageCircle className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">Sélectionnez une conversation pour commencer</p>
              </div>
            </CardContent>
          )}
        </Card>
      </div>
    </div>
  )
}
