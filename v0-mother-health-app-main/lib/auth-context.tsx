"use client"

import type React from "react"
import { createContext, useContext, useState, useEffect } from "react"

export type UserRole = "mother" | "doctor" | "child"

export interface User {
  id: string
  name: string
  email: string
  role: UserRole
}

interface AuthContextType {
  user: User | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  signup: (name: string, email: string, password: string, role: UserRole) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  // Initialize from localStorage on mount
  useEffect(() => {
    const storedUser = localStorage.getItem("familyhealth_user")
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser))
      } catch (error) {
        console.error("Failed to parse stored user:", error)
        localStorage.removeItem("familyhealth_user")
      }
    }
    setLoading(false)
  }, [])

  const login = async (email: string, password: string) => {
    setLoading(true)
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000))

      // Mock user data - in production, this would come from your backend
      const mockUser: User = {
        id: "user_" + Math.random().toString(36).substr(2, 9),
        name: email.split("@")[0],
        email,
        role: "mother",
      }

      setUser(mockUser)
      localStorage.setItem("familyhealth_user", JSON.stringify(mockUser))
    } finally {
      setLoading(false)
    }
  }

  const signup = async (name: string, email: string, password: string, role: UserRole) => {
    setLoading(true)
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000))

      const newUser: User = {
        id: "user_" + Math.random().toString(36).substr(2, 9),
        name,
        email,
        role,
      }

      setUser(newUser)
      localStorage.setItem("familyhealth_user", JSON.stringify(newUser))
    } finally {
      setLoading(false)
    }
  }

  const logout = () => {
    setUser(null)
    localStorage.removeItem("familyhealth_user")
  }

  return <AuthContext.Provider value={{ user, loading, login, signup, logout }}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}
