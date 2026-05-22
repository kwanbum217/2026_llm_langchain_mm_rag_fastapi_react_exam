#OPEN AI 채팅 모델을 LanChain 방식으로 사용할 수 있도록 하는 클래스
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
#프롬프트를 템플릿으로 관리할 수 있도록 하는 클래스
from langchain_core.prompts import ChatPromptTemplate
#LLM이 반환한 json 문자열을 파이썬 딕셔너리를 변환하는 클래스 
# StrOutputParser : 문자열을 그대로 문자열로.. (보고서, 요약 , 글쓰기)
# JsonOutputParser : json 문자열을 파이썬 딕셔너리로
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# [1] 프롬프트 템플릿 만들기 
# {detections_text} 부분에 invoke() 할 때 실제 탐지 결과 문자열이 들어가게 된다 
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 보안 전문가입니다. 반드시 JSON으로만 응답하세요."), # 페르소나 주입, 응답형식 결정
    ("human", "탐지 결과:\n{detections_text}\n\n위험도, 이유, 조치사항을 JSON으로 반환하세요."), # 사용자 실제 요청 메세지
])

# [2] 모델 만들기 (사용할 llm 모델 지정), ChatOpenAI open ai api를 랭체인 방식으로 사용할 수 있도록 하는 클래스
# temperature : (0 ~ 2)의 실수, 0 :가장 일관적(분석/분류 적합, 채점), 0.3 : 약간 유연(요약, 보고서) 0.7 이상 : 창의적(아이디어, 스토리)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# [3] 출력 파서 만들기 , LLM이 반환한 JSON 문자열을 파이썬 딕셔너리로 변환하는 클래스
json_parser = JsonOutputParser()

# [4] Chain으로 연결하기 
# LCEL(LangChain Expression Language) : 랭체인에서만 사용하는 전용 연산자 '|'
# 데이터가 흐르는 방향을 표시 (왼쪽에서 오른쪽으로 데이터가 흘러간다)
# analysis_chain = prompt | llm | json_parser
# 입력 데이터
#    ↓
# PromptTemplate이 메시지를 만든다
#    ↓
# LLM이 응답한다
#    ↓
# OutputParser가 응답을 딕셔너리로 바꾼다
analysis_chain = prompt | llm | json_parser

# [5] langchain 체인 실행하기
result = analysis_chain.invoke({
    "detections_text": "- person 탐지 (신뢰도 91%)\n- car 탐지 (신뢰도 95%)"
})

# 출력
print(result["risk_level"])
print(result["reason"])
print(result["action"])
