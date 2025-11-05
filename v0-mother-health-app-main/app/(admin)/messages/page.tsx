"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Search } from "lucide-react"

interface PatientMessage {
  id: string
  patientName: string
  lastMessage: string
  timestamp: string
  unread: boolean
}

const messages: PatientMessage[] = [
  {
    id: "1",
    patientName: "Emma Dupont (Mère)",
    lastMessage: "Merci pour vos conseils, Emma va mieux",
    timestamp: "Aujourd'hui 14:30",
    unread: true,
  },
  {
    id: "2",
    patientName: "Lucas Martin (Mère)",
    lastMessage: "Quand pouvons-nous avoir une consultation?",
    timestamp: "Hier 10:15",
    unread: false,
  },
]

export default function MessagesPage() {
  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Messages</h1>
        <p className="text-muted-foreground mt-2">Gérez vos communications avec les patients</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Conversations</CardTitle>
          <div className="mt-4 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input placeholder="Rechercher une conversation..." className="pl-10" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className="p-4 border border-border rounded-lg hover:bg-muted/50 transition-colors cursor-pointer"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="font-medium text-foreground">{msg.patientName}</p>
                    <p className="text-sm text-muted-foreground mt-1">{msg.lastMessage}</p>
                  </div>
                  {msg.unread && <div className="w-2 h-2 bg-primary rounded-full mt-1" />}
                </div>
                <p className="text-xs text-muted-foreground mt-2">{msg.timestamp}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
