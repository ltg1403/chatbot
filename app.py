from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# 챗봇 초기화
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"response": "메시지를 입력해주세요!"})
    
    response = chatbot(user_message)
    chatbot_reply = response[0]["generated_text"]
    return jsonify({"response": chatbot_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

