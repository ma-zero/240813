import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일을 읽어옵니다
header = ['O2', 'CO2', 'CO', 'H2S', 'HC']
file_path = 'team6-data.csv'  # CSV 파일의 경로를 입력하세요
df = pd.read_csv('./projectdata/team6-data.csv', names=header )

# 데이터 확인 (시간 정보가 포함되어 있는지 확인)
print(df.head())

# 데이터가 시간 정보와 농도 수치를 포함하고 있어야 합니다
# 여기서는 '시간'이라는 컬럼이 시간 정보를 포함한다고 가정합니다.
# 시간 컬럼을 인덱스로 설정
df['h'] = pd.to_datetime(df['h'])  # '시간' 컬럼을 datetime 형식으로 변환 (필요시)

# 밀도 플롯을 저장할 디렉토리를 생성합니다
import os
if not os.path.exists('./results'):
    os.makedirs('./results')

# 밀도 플롯 생성
plt.figure(figsize=(12, 10))
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, ax=plt.gca())
plt.suptitle('Density Plots of Gas Concentrations')
plt.savefig('./results/density.png')  # 결과를 파일로 저장

# 시간에 따른 농도 수치 플롯 생성
plt.figure(figsize=(12, 10))

for column in df.columns:
    if column != 'h':  # 시간 컬럼을 제외하고 농도 수치 플롯을 생성
        plt.plot(df['h'], df[column], label=column)

plt.title('Concentration over Time')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.savefig('./results/concentration_over_time.png')  # 결과를 파일로 저장

# 플롯을 화면에 표시 (선택 사항)
plt.show()
