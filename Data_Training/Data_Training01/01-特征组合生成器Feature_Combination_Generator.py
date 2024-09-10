from itertools import combinations


 # 'Longitude', 'Latitude', 'Magnitude', 'Depth', 'StationLongitude', 'StationLatitude', 'StationElevation', 'Azimuth', 'DistanceToDam'
# features = ['Longitude', 'Latitude', 'Magnitude', 'Depth', 'DistanceToDam', 'StationLongitude', 'StationLatitude', 'StationElevation', 'Azimuth']
# features = ['TriggerTime', 'ShockTime', 'Time','Longitude', 'Latitude', 'Magnitude', 'Depth', 'DistanceToDam', 'StationLongitude', 'StationLatitude', 'StationElevation', 'Azimuth']

# , 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'
features = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']

# 生成所有可能的组合
all_combinations = []
for r in range(1, 10):  # 从2到9
    combinations_r = list(combinations(features, r))
    all_combinations.extend(combinations_r)

# 将组合转换成你要求的格式
formatted_combinations = [list(combo) for combo in all_combinations]

# 打印结果
for i, combo in enumerate(formatted_combinations, 1):
    print(f"  {combo},")
