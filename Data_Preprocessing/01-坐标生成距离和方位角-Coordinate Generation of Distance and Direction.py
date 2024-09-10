import pandas as pd
from geopy.distance import geodesic
import math

# 读取CSV文件  E:\SCI_Part\A_DataBase\A_Raw
# data = pd.read_csv("H:\C2_SchlProjectFiles\P2_SeismicDataPrediction\SCI_Part\A_DataBase/way.csv")
# data = pd.read_csv("H:\C2_SchlProjectFiles\P2_SeismicDataPrediction\SCI_Part\A_DataBase\Kik-Net&K-Net/2023Train.csv")


# 读取CSV文件
data = pd.read_csv("XGB预测2020.csv")


# 计算距离和方向
def calculate_distance(coord1, coord2):
    return round(geodesic(coord1, coord2).kilometers, 5)

def calculate_direction(coord1, coord2):
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    angle = math.atan2(y, x)
    degrees = math.degrees(angle)
    if degrees < 0:
        degrees += 360
    return round(degrees, 2)

# # 添加距离和方向列
# data["距离(km)"] = data.apply(lambda row: calculate_distance((row["Latitude"], row["Longitude"]),
#                                                            (row["StationLatitude"], row["StationLongitude"])), axis=1)
# 添加距离和方向列
data["DistanceToDam(km)"] = data.apply(lambda row: calculate_distance((row["Latitude"], row["Longitude"]),
                                                           (row["StationLatitude"], row["StationLongitude"])), axis=1)



# data["方向(°)"] = data.apply(lambda row: calculate_direction((row["Latitude"], row["Longitude"]),
#                                                            (row["StationLatitude"], row["StationLongitude"])), axis=1)

data["Azimuth(°)"] = data.apply(lambda row: calculate_direction((row["Latitude"], row["Longitude"]),
                                                           (row["StationLatitude"], row["StationLongitude"])), axis=1)


# 保存到新的CSV文件
# data.to_csv("H:\C2_SchlProjectFiles\P2_SeismicDataPrediction\SCI_Part\A_DataBase\Kik-Net&K-Net/坐标_with_distance2023KNet_.csv", index=False)


# 保存到新的CSV文件，使用相对路径
data.to_csv("XGB预测20201.csv", index=False)