import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일을 읽어옵니다
header = ['O2', 'CO2', 'CO', 'H2S', 'HC']
file_path = 'team6-data.csv'  # CSV 파일의 경로를 입력하세요
df = pd.read_csv('./projectdata/team6-data.csv', names=header)

# 데이터 확인
print(df.head())

# 밀도 플롯을 저장할 디렉토리를 생성합니다
import os
if not os.path.exists('./results'):
    os.makedirs('./results')

# 밀도 플롯 생성
plt.clf()  # 이전의 플롯을 지우고 새로 시작
df.plot(kind='density', figsize=(12, 10), subplots=True, layout=(3, 3), sharex=False)
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.suptitle('Density Plots of Gas Concentrations')
plt.savefig('./results/density.png')  # 결과를 파일로 저장


# 플롯을 화면에 표시 (선택 사항)
plt.show()
