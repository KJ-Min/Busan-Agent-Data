import pandas as pd
import numpy as np
from openai import OpenAI
import time
from dotenv import load_dotenv
import os

# API 키 정보 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI()

def summarize_content(markdown_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
                주어진 마크다운 텍스트에서 제공된 정보만을 사용하여 식당 정보를 요약해주세요.
                - 제공된 텍스트에 없는 정보는 절대 추가하지 마세요
                - 실제 언급된 정보만 포함하세요
                - 3줄 이내의 간단한 문장으로 작성해주세요
                """},
                {"role": "user", "content": markdown_text}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        summary = response.choices[0].message.content.strip()
        return f"요약: {summary}\n\n{markdown_text}"
    
    except Exception as e:
        print(f"Error during summarization: {e}")
        return markdown_text

def main():
    # Restaurant/BFST
    BFTS_input_path = '/home/ubuntu/KJ-Min/Data/Restaurant/BFTS/BFTS_4_MarkdownOnly.csv'
    BFTS_output_path = '/home/ubuntu/KJ-Min/Data/Restaurant/BFTS/BFTS_5_Summarized_Markdown.csv'
    _7B_input_path = '/home/ubuntu/KJ-Min/Data/Restaurant/7B/7B_4_MarkdownOnly.csv'
    _7B_output_path = '/home/ubuntu/KJ-Min/Data/Restaurant/7B/7B_5_Summarized_Markdown.csv'
    
    # 청크 크기 설정
    CHUNK_SIZE = 100
    
    # CSV 파일을 청크로 읽기
    for chunk_num, chunk_df in enumerate(pd.read_csv(_7B_input_path, chunksize=CHUNK_SIZE)):
        print(f"Processing chunk {chunk_num + 1}")
        
        # 각 청크의 행 처리
        for idx in chunk_df.index:
            print(f"Processing row {idx} in chunk {chunk_num + 1}")
            
            # 마크다운 내용 가져오기
            content = chunk_df.loc[idx, 'markdown_content']
            
            # 요약 생성 및 추가
            updated_content = summarize_content(content)
            
            # 결과 저장
            chunk_df.loc[idx, 'markdown_content'] = updated_content
            
            # API 호출 제한을 위한 대기
            time.sleep(1)
        
        # 각 청크를 처리한 후 바로 파일에 저장 (mode='a'는 첫 청크 이후 추가)
        mode = 'w' if chunk_num == 0 else 'a'
        header = True if chunk_num == 0 else False
        
        chunk_df.to_csv(_7B_output_path, 
                       mode=mode, 
                       header=header, 
                       index=False, 
                       encoding='utf-8-sig')
        
        print(f"Chunk {chunk_num + 1} completed and saved")

    print("All processing completed!")

if __name__ == "__main__":
    main() 