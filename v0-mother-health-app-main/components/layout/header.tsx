"use client"
import { Menu, Bell, User } from "lucide-react"

interface HeaderProps {
  onMenuClick: () => void
}

export function Header({ onMenuClick }: HeaderProps) {
  return (
    <header className="h-16 border-b border-border bg-card flex items-center justify-between px-6">
      <button onClick={onMenuClick} className="md:hidden text-foreground hover:bg-muted rounded-lg p-2">
        <Menu className="w-5 h-5" />
      </button>

      <div className="flex-1" />

      <div className="flex items-center gap-4">
        <button className="text-foreground hover:bg-muted rounded-lg p-2 relative">
          <Bell className="w-5 h-5" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-accent rounded-full" />
        </button>
        <button className="text-foreground hover:bg-muted rounded-lg p-2">
          <User className="w-5 h-5" />
        </button>
      </div>
    </header>
  )
}
