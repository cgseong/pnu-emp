# ModuleNotFoundError 문제 진단

**에러 메시지**:
```
ModuleNotFoundError: This app has encountered an error...
File "/mount/src/pnu-emp/streamlit_app.py", line 26, in <module>
    from config import init_supabase
```

**에러 발생 위치**: streamlit_app.py 라인 26 (config 모듈 임포트)

---

## 🔍 **근본 원인 분석**

### 문제 1: config 모듈을 찾을 수 없음

**가능한 이유들**:

1. **config.py 파일이 GitHub에 푸시되지 않음**
   - 로컬에는 있지만 원격 저장소에 없음
   - Streamlit Cloud가 배포할 때 config.py를 찾을 수 없음

2. **`__pycache__` 파일이 커밋됨**
   - Python의 캐시 파일이 git에 추적되고 있을 수 있음
   - 환경 간 호환성 문제 발생

3. **파일 경로 문제**
   - Streamlit Cloud의 Linux 환경과 로컬 Windows 환경의 경로 불일치
   - 하지만 config.py는 같은 디렉토리에 있으므로 이 가능성은 낮음

### 문제 2: Streamlit Cloud 환경이 완전히 초기화되지 않음

- 앱 시작 시점에 모든 모듈이 준비되지 않았을 수 있음
- Python 환경 설정 미완료

---

## ✅ **진단 결과**

### 확인된 사항

```
로컬 환경:
  ✓ config.py 파일 존재
  ✓ streamlit_app.py 파일 존재
  ✓ 임포트 에러 없음
  ✓ 모든 함수 동작

Streamlit Cloud:
  ✗ config.py를 찾을 수 없음
  → GitHub에 푸시되지 않았거나
  → 배포 중에 누락되었을 가능성 높음
```

---

## 🔧 **가장 가능성 높은 해결 방법**

### **해결책 1: GitHub에서 현재 상태 확인** (무조건 필요)

```bash
# 1. GitHub 저장소 확인
# pnu-emp 저장소의 파일 목록을 확인하세요:
# https://github.com/[your-username]/pnu-emp/tree/main

# 확인 사항:
# □ config.py가 보이는가?
# □ streamlit_app.py가 보이는가?
# □ requirements.txt가 보이는가?
```

**결과에 따라**:
- **config.py가 없으면** → 아래의 "해결책 2" 실행
- **config.py가 있으면** → "해결책 3" 실행

---

### **해결책 2: config.py 파일을 GitHub에 푸시** (가장 가능성 높음)

```bash
# 1. config.py가 추적되지 않는지 확인
git status

# 2. config.py를 스테이징 에어리어에 추가
git add config.py

# 3. 커밋
git commit -m "Add: config.py for Supabase initialization"

# 4. GitHub에 푸시
git push origin main
```

**푸시 후**:
- Streamlit Cloud가 자동으로 감지 (몇 초)
- 새 버전 배포 시작 (1-3분)
- 앱 자동 재로드

---

### **해결책 3: 강제 재배포** (config.py가 이미 있는 경우)

Streamlit Cloud에서:

1. **앱 설정 (Manage app)** 클릭
2. **Reboot** 버튼 클릭
3. **2-5분 대기**
4. 앱 상태 확인

---

## 📋 **확인해야 할 체크리스트**

### Step 1: 로컬 확인
```bash
cd c:\Users\SW40904\OneDrive\Code\streamlit\emp

# 파일 확인
ls -la config.py              # config.py 존재하는가?
ls -la streamlit_app.py       # streamlit_app.py 존재하는가?
ls -la requirements.txt        # requirements.txt 존재하는가?

# Git 상태 확인
git status                    # config.py가 추적되는가?
git log --oneline -5          # 최근 커밋 확인
```

### Step 2: GitHub 확인
```
https://github.com/[your-username]/pnu-emp
```

다음 파일들이 보이는가?
- [ ] config.py
- [ ] streamlit_app.py
- [ ] requirements.txt
- [ ] .streamlit/secrets.toml
- [ ] .streamlit/config.toml

### Step 3: Streamlit Cloud 확인
```
https://share.streamlit.io
```

앱 상태 확인:
- [ ] 앱이 보이는가?
- [ ] "Your app is running" 표시되는가?
- [ ] 빨간 에러 표시되는가?

---

## 📝 **다음 조치 결정 흐름도**

```
┌─ GitHub에 config.py가 있는가?
│
├─ [YES] → GitHub 확인 후 "해결책 3: 강제 재배포" 실행
│          (Streamlit Cloud에서 Reboot)
│
└─ [NO]  → "해결책 2: config.py 푸시" 실행
           git add config.py
           git commit -m "Add config.py"
           git push origin main
```

---

## 🎯 **확인 후 선택**

**다음 중 하나를 선택하세요:**

### 옵션 A: "GitHub에 config.py가 없습니다"
→ 제가 GitHub에 푸시하는 명령어를 정확히 알려드리겠습니다.

### 옵션 B: "GitHub에 config.py가 있습니다"
→ Streamlit Cloud에서 Reboot하는 방법을 알려드리겠습니다.

### 옵션 C: "GitHub를 확인할 수 없습니다"
→ 현재 로컬 상황을 알려주면 진단하겠습니다.

---

## 💡 **추가 팁**

### 만약 위의 방법들이 작동하지 않으면:

```bash
# 1. 전체 상태 확인
git status
git log --oneline -5

# 2. config.py의 위치 확인
pwd
ls -la config.py

# 3. 강제 푸시 (마지막 수단)
git add .
git commit -m "Fix: Ensure all files are committed"
git push origin main
```

---

**다음**: 위의 체크리스트를 확인하시고 선택지를 알려주세요.
