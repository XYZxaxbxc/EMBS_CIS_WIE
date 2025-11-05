import { analyzeMentalHealth } from "@/lib/ai-utils"

export async function POST(request: Request) {
  try {
    const { message, conversationHistory } = await request.json()

    if (!message) {
      return Response.json({ error: "Message is required" }, { status: 400 })
    }

    const result = await analyzeMentalHealth(message, conversationHistory)

    if (!result.success) {
      return Response.json({ error: result.response }, { status: 500 })
    }

    return Response.json({ response: result.response })
  } catch (error) {
    console.error("Chat API error:", error)
    return Response.json({ error: "Failed to process message" }, { status: 500 })
  }
}
