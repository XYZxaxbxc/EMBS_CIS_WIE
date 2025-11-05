const BACKEND_URL =
  process.env.BACKEND_URL ??
  process.env.NEXT_PUBLIC_BACKEND_URL ??
  "http://127.0.0.1:8000";
function buildBackendUrl(path: string) {
  return `${BACKEND_URL.replace(/\/$/, "")}${path}`;
}
export async function POST(request: Request) {
  try {
    const form = await request.formData();
    const file = form.get("image");
    if (!file || !(file instanceof File)) {
      return Response.json({ error: "Image file is required" }, { status: 400 });
    }
    const portionEntry = form.get("portion_g");
    const topKEntry = form.get("k");
    const portion =
      typeof portionEntry === "string" && portionEntry.trim() !== "" ? portionEntry : "200";
    const topK = typeof topKEntry === "string" && topKEntry.trim() !== "" ? topKEntry : "5";
    const backendForm = new FormData();
    backendForm.append("image", file, file.name || "upload.jpg");
    backendForm.append("portion_g", portion);
    backendForm.append("k", topK);
    const backendResponse = await fetch(buildBackendUrl("/api/food"), {
      method: "POST",
      body: backendForm,
    });
    const contentType = backendResponse.headers.get("content-type") ?? "";
    if (!backendResponse.ok) {
      const errorBody = contentType.includes("application/json")
        ? await backendResponse.json()
        : { detail: await backendResponse.text() };
      console.error("Food model backend error:", backendResponse.status, errorBody);
      return Response.json(
        { error: errorBody?.detail || "Food recognition failed" },
        { status: backendResponse.status },
      );
    }
    const data = contentType.includes("application/json")
      ? await backendResponse.json()
      : await backendResponse.text();
    return Response.json(data);
  } catch (error) {
    console.error("Food analysis proxy error:", error);
    return Response.json({ error: "Unable to analyze meal image" }, { status: 500 });
  }
}
