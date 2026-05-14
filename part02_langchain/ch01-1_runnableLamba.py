from langchain_core.runnables import RunnableLambda


def add (text) :
  return text + "!"

def make_intro(data) : 
  return f"{data['name']}님은 {data['age']}세입니다."

# RunnableLambda는 일반 파이썬 함수를 LangChain 파이프라인에 끼워넣기 위한 어댑터이다.
# 함수의 주소를 보관하고 있다가 체인(파이프라인)에 포함시켜 실행시켜준다. 
# 여러개의 함수도 실행 시킬수 있다

# RunnableLambda의 기본 활용법
chain = RunnableLambda(add)
result = chain.invoke("안녕하세요")
print(result)


# RunnableLambda를 통해 딕셔너리 입력 가능
intro_chian = RunnableLambda(make_intro)
result2 = intro_chian.invoke({
  "name":"둘리", 
  "age":25
})

print(result2)

# RunnableLambda를 여러개 연속으로 연결 할 수 있다.(중요!)

def remove_spaces (text) : 
  return text.strip()

def add_prefix (text) : 
  return f"입력값 : {text}"

multiChain = (
  RunnableLambda(remove_spaces) | RunnableLambda(add_prefix)
)

result3 = multiChain.invoke("   LangChain   ")
print(result3)
