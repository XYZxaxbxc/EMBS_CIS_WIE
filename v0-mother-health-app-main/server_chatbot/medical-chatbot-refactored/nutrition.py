from groq import Groq
import os

# --- Configuration du client ---
# ⚠️ Bonne pratique : mets ta clé dans une variable d'environnement
# Exemple dans ton terminal : setx MEFTEH "ta_cle_ici"
client = Groq(api_key=os.getenv("MEFTEH"))

# --- Fonction du nutritionniste IA ---
def nutritionist_advice(user_diet, ingredients=None):
    """Analyse le régime et propose un plan alimentaire + recettes équilibrées."""

    system_prompt = (
        "Tu es un nutritionniste professionnel et empathique. "
        "Ton rôle est :\n"
        "- D'analyser le régime alimentaire fourni par l'utilisateur.\n"
        "- De proposer un plan nutritionnel équilibré adapté à ses besoins et objectifs (santé, perte de poids, énergie...).\n"
        "- De suggérer des recettes simples et équilibrées selon les ingrédients disponibles.\n"
        "Tes réponses doivent être structurées avec des sections claires, en français simple et compréhensible."
    )

    # Message de l'utilisateur
    user_message = f"Voici mon régime : {user_diet}"
    if ingredients:
        user_message += f"\nIngrédients disponibles : {', '.join(ingredients)}"

    # --- Création de la requête vers Groq ---
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.9,
        max_completion_tokens=1500,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
    )

    # --- Affichage du texte généré en streaming ---
    print("\n🍏 Réponse du nutritionniste :\n")
    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="", flush=True)


# --- Exemple d’utilisation ---
if __name__ == "__main__":
    diet = "Je mange souvent du pain, du fromage, et peu de légumes. Je bois aussi beaucoup de soda."
    ingredients = ["poulet", "riz", "tomates", "avocat", "œufs", "laitue"]
    nutritionist_advice(diet, ingredients)
