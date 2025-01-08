import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)  # Flask 애플리케이션 생성
CORS(app)  # 모든 도메인에서 오는 요청을 허용

# 작은 모델을 사용
model_name = "distilgpt2"
# AutoTokenizer와 AutoModelForCausalLM을 사용하여 모델과 토크나이저를 로드: pipeline을 사용할 때 model과 tokenizer를 명시적으로 전달.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # pad_token을 eos_token으로 설정

chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route("/")  # 홈페이지 라우팅
def home():
    return render_template("index.html")  # index.html 템플릿을 렌더링하여 반환

@app.route("/chat", methods=["POST"])  # POST 요청을 받는 '/chat' 라우팅
def chat():
    # 클라이언트에서 보낸 메시지 추출 (JSON 형태로)
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"response": "메시지를 입력해주세요!"})

    # 모델을 사용하여 사용자 메시지에 대한 응답 생성
    # max_length: 생성할 텍스트의 최대 길이, num_return_sequences: 생성할 텍스트 개수
    inputs = tokenizer(user_message, return_tensors="pt", truncation=True, max_length=50, padding="max_length")
    
    # attention_mask를 설정
    attention_mask = inputs["attention_mask"]

    # 모델로부터 응답 생성
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1, 
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                             attention_mask=attention_mask)

    # 응답 디코딩
    chatbot_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 생성된 응답을 JSON 형식으로 반환
    return jsonify({"response": chatbot_reply})

# Flask 실행 설정
if __name__ == "__main__":  # 이 파일이 메인으로 실행될 때만 아래 코드 실행
    # 호스트는 '0.0.0.0'으로 설정하여 외부에서 접근할 수 있도록 하고,
    # 포트 번호는 Render에서 제공하는 'PORT' 환경 변수를 사용하며, 기본값은 5000
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
