# Git 저장소 설정 - 단계별 가이드

**현재 상황**: 로컬 폴더가 Git 저장소로 초기화되지 않음
**해결 방법**: Git 저장소 연결 및 config.py 푸시

---

## 📋 **실행할 명령어** (순서대로)

### Step 1: Git 저장소 초기화 및 리모트 연결

**Windows PowerShell에서 다음을 실행하세요:**

```powershell
cd c:\Users\SW40904\OneDrive\Code\streamlit\emp
```

### Step 2: 기존 리모트 저장소 확인

```powershell
git remote -v
```

**결과**:
- 아무것도 나오지 않으면 → Step 3 실행
- `origin ... github.com ...`이 나오면 → Step 4로 이동

### Step 3: 리모트 저장소 추가 (GitHub 저장소가 있는 경우)

```powershell
git remote add origin https://github.com/YOUR-USERNAME/pnu-emp.git
```

**주의**: `YOUR-USERNAME`을 실제 GitHub 사용자명으로 변경하세요

**예시**:
```powershell
git remote add origin https://github.com/sw40904/pnu-emp.git
```

### Step 4: Git 설정 확인

```powershell
git config --list | Select-String user
```

**결과**:
- `user.name`과 `user.email`이 보여야 함
- 없으면 다음을 실행:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 5: 모든 파일을 스테이징 에어리어에 추가

```powershell
git add .
```

### Step 6: 상태 확인

```powershell
git status
```

**확인 사항**:
```
On branch main (또는 master)
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)
        new file:   config.py
        ...
```

### Step 7: 커밋 생성

```powershell
git commit -m "Add config.py for Supabase initialization"
```

### Step 8: GitHub에 푸시

```powershell
git push -u origin main
```

**만약 branch 이름이 다르면**:
```powershell
git push -u origin master
```

---

## 🔍 **혹시 모를 경우**

### 현재 상황이 복잡하면 - 처음부터 새로 설정

```powershell
# 1. 기존 리모트 제거 (있으면)
git remote remove origin

# 2. 새로운 리모트 추가
git remote add origin https://github.com/YOUR-USERNAME/pnu-emp.git

# 3. 현재 branch 확인
git branch -a

# 4. main branch로 이동 (있으면)
git checkout main

# 5. GitHub에서 최신 코드 가져오기
git pull origin main

# 6. 모든 파일 추가
git add .

# 7. 커밋
git commit -m "Update: Add missing files including config.py"

# 8. 푸시
git push origin main
```

---

## ⚠️ **주의사항**

### Git 사용자 정보가 필요한 경우

```powershell
git config --global user.name "Your GitHub Username"
git config --global user.email "your-email@github.com"
```

### GitHub 인증 문제가 발생하면

1. **Personal Access Token 사용** (권장):
   - GitHub → Settings → Developer settings → Personal access tokens
   - "Generate new token" 클릭
   - "repo" 범위 선택
   - Token 복사
   - Push 시 비밀번호 대신 Token 입력

2. **SSH 키 사용** (고급):
   - GitHub의 SSH 키 설정 가이드 참조

---

## 📝 **완료 체크리스트**

실행 후 다음을 확인하세요:

- [ ] `git status` 실행 후 "clean working directory" 표시됨
- [ ] `git log --oneline -5` 실행 후 방금 생성한 커밋 보임
- [ ] GitHub 웹사이트에서 config.py 파일 보임
- [ ] Streamlit Cloud에서 자동 배포 시작됨 (로그에 나타남)

---

## 🚀 **푸시 후 예상 결과**

### Streamlit Cloud에서:

1. **자동 감지** (몇 초)
   - GitHub 푸시 감지

2. **자동 배포** (1-3분)
   - 로그에 "Building" 표시
   - 파일 다운로드 및 설치

3. **자동 재로드** (완료 후)
   - 앱이 새로 로드됨
   - "Your app is running" 표시

4. **Secrets 설정 확인**
   - Settings → Secrets
   - [supabase] 섹션 존재하는지 확인

---

## 다음 단계

1. **위의 Step 1-8을 실행**
2. **GitHub에서 config.py 확인**
3. **Streamlit Cloud의 로그 확인** (약 2-3분 소요)
4. **앱이 정상 로드되는지 확인**

문제 발생 시 에러 메시지를 알려주세요!

