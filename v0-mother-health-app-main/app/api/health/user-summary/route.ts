import { generateText } from "ai"

export async function POST(request: Request) {
  try {
    const { userId, mentalHealthData, nutritionData } = await request.json()

    const { text } = await generateText({
      model: "openai/gpt-4-mini",
      system: `You are a health summary generator for a family health platform.
Create concise, actionable health summaries based on user data.
Speak French and respond in French.
Be positive and encouraging.`,
      prompt: `Generate a health summary for user ${userId} with mental health data: ${JSON.stringify(mentalHealthData)} 
and nutrition data: ${JSON.stringify(nutritionData)}. Include key insights and action items.`,
    })

    return Response.json({ summary: text })
  } catch (error) {
    console.error("User summary API error:", error)
    return Response.json({ error: "Failed to generate summary" }, { status: 500 })
  }
}
