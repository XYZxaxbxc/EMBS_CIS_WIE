export async function GET() {
  try {
    return Response.json({
      status: "healthy",
      message: "FamilyHealth API is running",
      version: "1.0.0",
      services: {
        mentalHealth: "active",
        nutrition: "active",
        messaging: "active",
        analytics: "active",
      },
    })
  } catch (error) {
    console.error("Health check error:", error)
    return Response.json({ status: "error", message: "Service unavailable" }, { status: 503 })
  }
}
