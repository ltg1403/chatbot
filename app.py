from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__) # Flask 애플리케이션 생성

# 작은 모델을 사용하여 챗봇 초기화
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")

@app.route("/") # 홈페이지 라우팅
def home():
    return render_template("index.html") # index.html 템플릿을 렌더링하여 반환

@app.route("/chat", methods=["POST"]) # POST 요청을 받는 '/chat' 라우팅
def chat():
    # 클라이언트에서 보낸 메시지 추출 (JSON 형태로)
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"response": "메시지를 입력해주세요!"})
    
    # 모델을 사용하여 사용자 메시지에 대한 응답 생성
    # max_length: 생성할 텍스트의 최대 길이, num_return_sequences: 생성할 텍스트 개수
    response = chatbot(user_message, max_length=50, num_return_sequences=1)
    chatbot_reply = response[0]["generated_text"] # 모델이 생성한 응답 추출
    
    # 생성된 응답을 JSON 형식으로 반환
    return jsonify({"response": chatbot_reply})

# Flask 실행 설정
if __name__ == "__main__": # 이 파일이 메인으로 실행될 때만 아래 코드 실행
    # 호스트는 '0.0.0.0'으로 설정하여 외부에서 접근할 수 있도록 하고,
    # 포트 번호는 Render에서 제공하는 'PORT' 환경 변수를 사용하며, 기본값은 5000
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
