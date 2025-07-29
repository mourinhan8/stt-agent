import google.generativeai as genai
import os

class LlmModel:
    def __init__(self):
        genai.configure(api_key="AIzaSyAb3f4kH8r4tuayBq61tlHAXH0xL-0GKjY")
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def get_response(self, prompt):
        try:
            # Gửi câu hỏi đến mô hình
            response = self.model.generate_content(prompt)

            return response.text

        except Exception as e:
            print(f"Đã xảy ra lỗi khi gọi API: {e}")
