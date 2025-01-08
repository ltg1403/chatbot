import os
import logging

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)  # Flask 애플리케이션 생성
CORS(app)  # 모든 도메인에서 오는 요청을 허용

# 로그 설정
logging.basicConfig(level=logging.DEBUG)

try:
    # 경량화된 모델로 설정
    # model_name = "sshleifer/tiny-gpt2"  
    model_name = "skt/kogpt2-distilled"
    
    # 토크나이저와 모델 로드
    if 'tokenizer' not in globals() or 'model' not in globals():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # pad_token을 eos_token으로 설정
    
    # 파이프라인 생성
    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("모델과 토크나이저가 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"모델 / 토크나이저 초기화 중 오류 발생: {e}")

@app.route("/")  # 홈페이지 라우팅
def home():
    try:
        return render_template("index.html")  # index.html 템플릿을 렌더링하여 반환
    except Exception as e:
        print(f"홈페이지 렌더링 중 오류 발생: {e}")
        return "서버 오류: 홈페이지를 로드할 수 없습니다.", 500

@app.route("/chat", methods=["POST"])  # POST 요청을 받는 '/chat' 라우팅
def chat():
    try:
        # 클라이언트에서 보낸 메시지 추출 (JSON 형태로)
        user_message = request.json.get("message")
        logging.debug(f"받은 메시지: {user_message}")
        
        if not user_message:
            return jsonify({"response": "메시지를 입력해주세요!"})
        
        # 모델을 사용하여 사용자 메시지에 대한 응답 생성
        try:
            inputs = tokenizer(user_message, return_tensors="pt", truncation=True, max_length=50, padding="max_length")
            attention_mask = inputs["attention_mask"]
            outputs = model.generate(
                inputs["input_ids"], 
                max_length=100, 
                num_return_sequences=1, 
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=attention_mask,
                no_repeat_ngram_size=2,  # 반복 방지
                do_sample=True,  # 샘플링을 사용
                top_p=0.9,  # top-p 샘플링
                top_k=50  # top-k 샘플링
            )
            chatbot_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.debug(f"생성된 응답: {chatbot_reply}")
        except Exception as e:
            print(f"모델 응답 생성 중 오류 발생: {e}")
            chatbot_reply = "응답을 생성하지 못했습니다."

        # 생성된 응답을 JSON 형식으로 반환
        return jsonify({"response": chatbot_reply})
    except Exception as e:
        print(f"채팅 처리 중 오류 발생: {e}")
        return jsonify({"response": "서버 오류로 인해 요청을 처리할 수 없습니다."}), 500

if __name__ == "__main__":
    try:
        # 호스트는 '0.0.0.0'으로 설정하여 외부에서 접근할 수 있도록 하고,
        # 포트 번호는 Render에서 제공하는 'PORT' 환경 변수를 사용하며, 기본값은 5000
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    except Exception as e:
        print(f"서버 실행 중 오류 발생: {e}")
