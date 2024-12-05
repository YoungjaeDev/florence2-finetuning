import gdown

# 공유 링크를 gdown에서 사용할 수 있는 URL로 변환
url = "https://drive.google.com/uc?id=1a9XB3r83ZCFWLOHBp8ooz3zQFl9rEIei"
output = "data.zip"

# 파일 다운로드
gdown.download(url, output, quiet=False)