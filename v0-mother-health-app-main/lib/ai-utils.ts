import { generateText, generateObject } from "ai"
import { z } from "zod"

// Mental Health Analysis Schema
export const mentalHealthAnalysisSchema = z.object({
  emotionalState: z.string().describe("Current emotional state"),
  concerns: z.array(z.string()).describe("Main concerns identified"),
  recommendations: z.array(z.string()).describe("Personalized recommendations"),
  urgency: z.enum(["low", "medium", "high"]).describe("Urgency level"),
})

// Nutrition Analysis Schema
export const nutritionAnalysisSchema = z.object({
  mealName: z.string().describe("Name of the meal"),
  calories: z.number().describe("Estimated calories"),
  macronutrients: z.object({
    protein: z.number(),
    carbs: z.number(),
    fat: z.number(),
  }),
  healthScore: z.number().min(0).max(10).describe("Health score 0-10"),
  recommendations: z.array(z.string()).describe("Nutritional recommendations"),
})

// Mental Health Chat
export async function analyzeMentalHealth(userMessage: string, conversationHistory: any[] = []) {
  try {
    const { text } = await generateText({
      model: "openai/gpt-4-mini",
      system: `You are a compassionate mental health support assistant for a family health platform.
You provide supportive, non-judgmental responses about mental health and emotional well-being.
You speak French and respond in French.
You are helpful for both parents and children.
Keep responses concise and empathetic.
If someone expresses serious mental health concerns, encourage them to seek professional help.
Always maintain confidentiality and privacy.`,
      prompt: userMessage,
    })

    return { success: true, response: text }
  } catch (error) {
    console.error("Mental health analysis error:", error)
    return {
      success: false,
      response: "Je suis désolé, je n'ai pas pu traiter votre message. Veuillez réessayer.",
    }
  }
}

// Nutrition Recipe Generation
export async function generateRecipe(query: string) {
  try {
    const { text } = await generateText({
      model: "openai/gpt-4-mini",
      system: `You are a professional nutritionist and recipe assistant for a family health platform.
You provide balanced, healthy recipes suitable for families with children.
You speak French and respond in French.
Always include nutritional information (calories, protein, carbs, fat).
Consider dietary restrictions and preferences.
Provide practical, easy-to-follow recipes.
Focus on healthy, age-appropriate meals for children.`,
      prompt: query,
    })

    return { success: true, response: text }
  } catch (error) {
    console.error("Recipe generation error:", error)
    return {
      success: false,
      response: "Je n'ai pas pu générer une recette. Veuillez réessayer.",
    }
  }
}

// Emotion Detection from Drawing (placeholder for future image analysis)
export async function analyzeDrawingEmotion(imageDescription: string) {
  try {
    const { object } = await generateObject({
      model: "openai/gpt-4-mini",
      schema: z.object({
        emotions: z.array(
          z.object({
            name: z.string(),
            score: z.number().min(0).max(1),
          }),
        ),
        summary: z.string(),
        recommendations: z.array(z.string()),
      }),
      prompt: `Analyze this child's drawing description and identify emotional indicators: ${imageDescription}`,
    })

    return { success: true, data: object }
  } catch (error) {
    console.error("Drawing emotion analysis error:", error)
    return { success: false, error: "Failed to analyze drawing" }
  }
}

// Health Recommendation Engine
export async function generateHealthRecommendations(
  mentalHealthScore: number,
  nutritionScore: number,
  userContext: string,
) {
  try {
    const { text } = await generateText({
      model: "openai/gpt-4-mini",
      system: `You are a health recommendation expert for a family health platform.
Provide personalized, actionable health recommendations based on mental health and nutrition scores.
Speak French and respond in French.
Be encouraging and supportive.
Focus on practical, achievable improvements.`,
      prompt: `Based on mental health score ${mentalHealthScore}/100 and nutrition score ${nutritionScore}/100, 
provide 3-5 personalized recommendations for: ${userContext}`,
    })

    return { success: true, recommendations: text }
  } catch (error) {
    console.error("Recommendation generation error:", error)
    return { success: false, error: "Failed to generate recommendations" }
  }
}
