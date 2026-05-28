import os
import sys
from supertonic import TTS

# Windows 인코딩 환경에서 출력 에러(UnicodeEncodeError) 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    print("=" * 60)
    print("[INFO] Supertonic 온디바이스 TTS 라이브러리 테스트를 시작합니다.")
    print("=" * 60)

    # 1. TTS 파이프라인 초기화
    print("[1/4] TTS 파이프라인 초기화 중...")
    tts = TTS()
    
    # 2. 보이스 스타일 선택 (M1~M5: 남성, F1~F5: 여성)
    # 기본 내장 목소리 중 M1(남성 1번)과 F1(여성 1번)을 가져옵니다.
    print("[2/4] 목소리 스타일(M1, F1) 로딩 중...")
    style_male = tts.get_voice_style("M1")
    style_female = tts.get_voice_style("F1")

    # 3. 합성할 텍스트 설정
    # CCTV 플랫폼 시나리오에 부합하는 긴급 상황 방송 및 안내 방송 텍스트 정의
    scenarios = [
        {
            "name": "warning_ko",
            "text": "경고합니다! 주차장 구역에서 거동이 수상한 외부인이 감지되었습니다. 보안 요원은 즉시 출동하여 현장을 확인하십시오.",
            "voice": style_male,
            "lang": "ko"
        },
        {
            "name": "emergency_en",
            "text": "Warning! Unidentified person detected in the parking structure. Security units, please respond immediately.",
            "voice": style_female,
            "lang": "en"
        },
        {
            "name": "normal_ko",
            "text": "안내 말씀 드립니다. 본 구역은 실시간 cctv 및 인공지능 분석을 통해 안전하게 모니터링되고 있습니다.",
            "voice": style_female,
            "lang": "ko"
        }
    ]

    # Waves 저장용 디렉토리 경로 지정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "waves")
    os.makedirs(output_dir, exist_ok=True)

    # 4. 음성 합성 및 저장 실행
    print("[3/4] 음성 합성 및 파일 저장 시작...")
    for idx, sc in enumerate(scenarios, 1):
        filename = f"supertonic_{sc['name']}.wav"
        filepath = os.path.join(output_dir, filename)
        
        print(f"\n({idx}/{len(scenarios)}) '{sc['name']}' 음성 생성 중...")
        print(f"   - 텍스트: \"{sc['text']}\"")
        print(f"   - 언어: {sc['lang']} | 목소리 스타일: {'M1(남성)' if sc['voice'] == style_male else 'F1(여성)'}")
        
        try:
            # 음성 합성 수행 (wav 오디오 배열과 재생 시간 반환)
            wav, duration = tts.synthesize(sc["text"], voice_style=sc["voice"], lang=sc["lang"])
            
            # 오디오 저장
            tts.save_audio(wav, filepath)
            print(f"   [성공] 음성 파일이 저장되었습니다. ({float(duration[0]):.2f}초)")
            print(f"   [경로] {filepath}")
            
        except Exception as e:
            print(f"   [오류] 음성 합성 실패: {e}")

    print("\n" + "=" * 60)
    print("[OK] Supertonic TTS 테스트 프로그램 실행이 완료되었습니다!")
    print("     waves 폴더 아래에서 생성된 supertonic_*.wav 파일들을 확인하세요.")
    print("=" * 60)

if __name__ == "__main__":
    main()
