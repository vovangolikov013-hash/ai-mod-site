from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re

app = Flask(__name__)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="blanchefort/rubert-base-cased-sentiment"
)

#generator = pipeline("text-generation", model="sberbank-ai/rugpt3medium_based_on_gpt2")
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")

def extract_film_title(generated_text: str) -> str:

    if not generated_text:
        return ""

    m = re.search(r'«([^»]{2,100})»', generated_text)
    if m:
        return m.group(1).strip()

    return generated_text.strip().splitlines()[0][:80].strip()

def generate_recommendation(mood):
    prompt = (f"Посоветуй только одну книгу для человека, у которого {mood} настроение."
              f" Напиши только название книги, без описаний и комментариев.")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=70,
        do_sample=True,
        top_p=0.7,
        temperature=0.5,
        num_return_sequences=1
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

@app.route("/", methods=["GET", "POST"])
def index():
    recommendation = ""
    user_text = ""

    if request.method == "POST":
        user_text = request.form["message"]

        result = sentiment_analyzer(user_text)[0]
        label = result["label"]

        if label == "POSITIVE":
            mood = "хорошее"
        elif label == "NEGATIVE":
            mood = "плохое"
        else:
            mood = "нейтральное"

        ai_text = extract_film_title(generate_recommendation(mood))
        recommendation = f"Настроение: {mood}.<br>Рекомендация: {ai_text}"

    return render_template("index.html", recommendation=recommendation, user_text=user_text)


if __name__ == "__main__":
    app.run(debug=True)
