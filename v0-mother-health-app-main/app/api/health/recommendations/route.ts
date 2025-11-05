import { generateHealthRecommendations } from "@/lib/ai-utils"

export async function POST(request: Request) {
  try {
    const { mentalHealthScore, nutritionScore, userContext } = await request.json()

    if (!mentalHealthScore || !nutritionScore || !userContext) {
      return Response.json({ error: "Missing required fields" }, { status: 400 })
    }

    const result = await generateHealthRecommendations(mentalHealthScore, nutritionScore, userContext)

    if (!result.success) {
      return Response.json({ error: result.error }, { status: 500 })
    }

    return Response.json({ recommendations: result.recommendations })
  } catch (error) {
    console.error("Health recommendations API error:", error)
    return Response.json({ error: "Failed to generate recommendations" }, { status: 500 })
  }
}
