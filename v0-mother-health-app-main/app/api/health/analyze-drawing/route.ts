import { analyzeDrawingEmotion } from "@/lib/ai-utils"

export async function POST(request: Request) {
  try {
    const { imageDescription } = await request.json()

    if (!imageDescription) {
      return Response.json({ error: "Image description is required" }, { status: 400 })
    }

    const result = await analyzeDrawingEmotion(imageDescription)

    if (!result.success) {
      return Response.json({ error: result.error }, { status: 500 })
    }

    return Response.json(result.data)
  } catch (error) {
    console.error("Drawing analysis API error:", error)
    return Response.json({ error: "Failed to analyze drawing" }, { status: 500 })
  }
}
