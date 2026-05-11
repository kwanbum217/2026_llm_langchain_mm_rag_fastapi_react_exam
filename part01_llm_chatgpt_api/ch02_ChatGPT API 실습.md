## 사전 준비 — API 키 발급 및 패키지 설치

### API 키 발급 순서

```
1. https://platform.openai.com 접속
2. 우측 상단 로그인 → [API Keys] 메뉴
3. [+ Create new secret key] 클릭
4. 키 이름 입력 (예: "cctv-project") → [Create secret key]
5. sk-proj-... 형태의 키 복사 (이 창을 닫으면 다시 못 봄!)
6. 결제 수단 등록 → Settings > Billing > Add payment method
   (소액 충전 권장: $5~10으로 이 강의 전체 실습 가능)
```

> ⚠️ **API 키 보안 주의사항**
> 
> - API 키는 **절대 코드에 직접 쓰지 않습니다**
> - `.env` 파일에 저장하고, `.gitignore`에 `.env` 추가 필수
> - 키가 유출되면 즉시 [API Keys] 페이지에서 삭제(Revoke)

