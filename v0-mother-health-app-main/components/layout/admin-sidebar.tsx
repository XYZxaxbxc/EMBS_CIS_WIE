"use client"

import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Users, BarChart3, MessageSquare, Settings, LogOut, X, Stethoscope } from "lucide-react"
import { useAuth } from "@/lib/auth-context"

interface SidebarProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function Sidebar({ open, onOpenChange }: SidebarProps) {
  const pathname = usePathname()
  const router = useRouter()
  const { logout, user } = useAuth()

  const menuItems = [
    { icon: Users, label: "Patients", href: "/admin/patients" },
    { icon: BarChart3, label: "Analyses", href: "/admin/analytics" },
    { icon: MessageSquare, label: "Messages", href: "/admin/messages" },
    { icon: Settings, label: "Paramètres", href: "/admin/settings" },
  ]

  const handleLogout = () => {
    logout()
    router.push("/login")
  }

  return (
    <>
      {/* Mobile overlay */}
      {open && <div className="fixed inset-0 bg-black/50 z-40 md:hidden" onClick={() => onOpenChange(false)} />}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed md:relative w-64 h-screen bg-sidebar border-r border-sidebar-border flex flex-col transition-transform duration-300 z-50 md:z-0",
          open ? "translate-x-0" : "-translate-x-full md:translate-x-0",
        )}
      >
        {/* Header */}
        <div className="p-6 border-b border-sidebar-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-sidebar-primary flex items-center justify-center">
              <Stethoscope className="w-5 h-5 text-sidebar-primary-foreground" />
            </div>
            <span className="font-bold text-sidebar-foreground"> Etma'en Pro</span>
          </div>
          <button
            onClick={() => onOpenChange(false)}
            className="md:hidden text-sidebar-foreground hover:bg-sidebar-accent rounded-lg p-1"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {menuItems.map((item) => {
            const Icon = item.icon
            const isActive = pathname.startsWith(item.href)
            return (
              <Link key={item.href} href={item.href}>
                <button
                  className={cn(
                    "w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors",
                    isActive
                      ? "bg-sidebar-primary text-sidebar-primary-foreground"
                      : "text-sidebar-foreground hover:bg-sidebar-accent",
                  )}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{item.label}</span>
                </button>
              </Link>
            )
          })}
        </nav>

        {/* User Info & Logout */}
        <div className="p-4 border-t border-sidebar-border space-y-3">
          {user && (
            <div className="px-2 py-2 text-sm">
              <p className="font-medium text-sidebar-foreground truncate">{user.name}</p>
              <p className="text-sidebar-foreground/60 text-xs capitalize">Médecin</p>
            </div>
          )}
          <Button
            onClick={handleLogout}
            variant="outline"
            className="w-full justify-start gap-2 text-sidebar-foreground border-sidebar-border hover:bg-sidebar-accent bg-transparent"
          >
            <LogOut className="w-4 h-4" />
            Déconnexion
          </Button>
        </div>
      </aside>
    </>
  )
}
